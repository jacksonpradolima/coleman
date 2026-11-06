import os
import subprocess
import sys
from datetime import datetime

import pytest

from coleman.artifacts import ArtifactWriter
from coleman.interchange.codec import decode_response
from coleman.interchange.contracts import InterchangeRequest, InterchangeResponse, accept_response
from coleman.interchange.publication import publish_request, publish_response

CHILD = """
import os, sys
from pathlib import Path
from coleman.artifacts import ArtifactWriter
from coleman.interchange.codec import decode_request, decode_response
from coleman.interchange.publication import publish_response
request=decode_request(Path(sys.argv[1]).read_bytes())
response=decode_response(Path(sys.argv[2]).read_bytes())
mode=sys.argv[4]
original=os.link
def link(source,destination,*args,**kwargs):
    if Path(destination).name=='response.json' and mode=='before':
        os._exit(91)
    result=original(source,destination,*args,**kwargs)
    if Path(destination).name=='response.json' and mode=='after':
        os._exit(92)
    return result
os.link=link
try:
    path=publish_response(request,response,ArtifactWriter(sys.argv[3]))
except ValueError:
    sys.exit(23)
print(path)
"""


def _arguments(tmp_path, request, response, mode="normal", name="producer"):
    request_path, response_path = tmp_path / f"{name}-request.json", tmp_path / f"{name}-response.json"
    request_path.write_text(request.to_json())
    response_path.write_text(response.to_json())
    return [sys.executable, "-c", CHILD, str(request_path), str(response_path), str(tmp_path / "artifacts"), mode]


def _environment():
    return {key: value for key, value in os.environ.items() if key not in {"PYTHONHOME", "PYTHONPATH"}}


def test_explicit_publication_idempotent_paths_and_actual_consumer_acceptance(tmp_path, coverage_pair):
    request, response = coverage_pair
    writer = ArtifactWriter(tmp_path)
    assert not list(tmp_path.iterdir())
    request_path = publish_request(request, writer)
    assert request_path.name == "request.json"
    assert not request_path.with_name("response.json").exists()
    response_path = publish_response(request, response, writer)
    assert response_path.read_text() == response.to_json()
    assert publish_response(request, response, writer) == response_path
    actual = decode_response(response_path.read_bytes())
    assert accept_response(request, actual, received_at=datetime.fromisoformat("2026-01-01T10:02:00Z")) == response
    assert len(list(tmp_path.rglob("response.json"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_same_logical_attempt_changed_payload_conflicts_not_new_file(tmp_path, coverage_pair, respond):
    request, response = coverage_pair
    writer = ArtifactWriter(tmp_path)
    path = publish_response(request, response, writer)
    changed = InterchangeRequest.model_validate(request.model_copy(update={"created_at": "2026-01-01T09:00:00Z"}))
    other = respond(changed, response.reply, produced_at=response.produced_at)
    assert changed.attempt_identity == request.attempt_identity
    with pytest.raises(ValueError, match="Conflicting"):
        publish_response(changed, other, writer)
    assert path.read_text() == response.to_json()
    assert len(list(tmp_path.rglob("request.json"))) == 1
    assert len(list(tmp_path.rglob("response.json"))) == 1
    altered = InterchangeResponse.model_validate(response.model_copy(update={"produced_at": "2026-01-01T10:02:00Z"}))
    with pytest.raises(ValueError, match="Conflicting"):
        publish_response(request, altered, writer)
    assert path.read_text() == response.to_json()


def test_explicit_new_attempt_separate_and_untrusted_labels_not_paths(tmp_path, coverage_pair, respond):
    request, response = coverage_pair
    changed = InterchangeRequest.model_validate(request.model_copy(update={"attempt_id": "../other-attempt"}))
    assert changed.logical_identity == request.logical_identity
    assert changed.attempt_identity != request.attempt_identity
    other = respond(changed, response.reply, produced_at=response.produced_at)
    writer = ArtifactWriter(tmp_path)
    first = publish_response(request, response, writer)
    second = publish_response(changed, other, writer)
    assert first.parent != second.parent
    assert second.resolve().is_relative_to(tmp_path.resolve())
    assert ".." not in second.relative_to(tmp_path).parts
    with pytest.raises(ValueError, match="retained producing request"):
        publish_response(request, other, writer)


@pytest.mark.parametrize("mode,code,completed", [("before", 91, False), ("after", 92, True)])
def test_real_process_interruption_never_exposes_partial_completed_reply(
    tmp_path, coverage_pair, mode, code, completed
):
    request, response = coverage_pair
    command = _arguments(tmp_path, request, response, mode)
    result = subprocess.run(command, env=_environment(), capture_output=True, text=True, timeout=20)
    assert result.returncode == code, result.stderr
    writer = ArtifactWriter(tmp_path / "artifacts")
    directory = writer.run_root / "interchange" / request.attempt_identity.removeprefix("sha256:")
    assert (directory / "request.json").exists()
    path = directory / "response.json"
    assert path.exists() is completed
    if completed:
        assert decode_response(path.read_bytes()) == response
    # Recovery supplies the retained host request/output, not executable state from a hash.
    assert publish_response(request, response, writer) == path
    assert accept_response(request, decode_response(path.read_bytes()), received_at=request.expires_at) == response


@pytest.mark.parametrize("conflicting", [False, True])
def test_real_concurrent_processes_identical_retries_or_one_conflict_winner(tmp_path, coverage_pair, conflicting):
    request, response = coverage_pair
    other = InterchangeResponse.model_validate(response.model_copy(update={"produced_at": "2026-01-01T10:02:00Z"}))
    commands = [
        _arguments(tmp_path, request, response, name="first"),
        _arguments(tmp_path, request, other if conflicting else response, name="second"),
    ]
    processes = [
        subprocess.Popen(command, env=_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for command in commands
    ]
    outputs = [process.communicate(timeout=20) for process in processes]
    codes = sorted(process.returncode for process in processes)
    assert codes == ([0, 23] if conflicting else [0, 0]), outputs
    paths = list((tmp_path / "artifacts").rglob("response.json"))
    assert len(paths) == 1
    actual = decode_response(paths[0].read_bytes())
    assert actual in (response, other)
    assert accept_response(request, actual, received_at=request.expires_at) == actual
    assert not list((tmp_path / "artifacts").rglob("*.tmp"))
