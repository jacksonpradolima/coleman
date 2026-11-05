"""Opaque v1 body framing preserves compatibility without decoding outcome values."""

import json
import random

import pytest

from coleman.application.workflow_import_header import _source_header


def test_header_projection_preserves_order_whitespace_and_escaped_opaque_values(monkeypatch):
    expected = {
        "schema_version": 1,
        "contract": "coleman-workflows-v1",
        "artifact_type": "coleman-workflow-evidence-v1",
        "plan": {"source": "declared", "units": "s", "values": [0, None, -1.0]},
    }
    original = json.JSONDecoder.raw_decode

    def guarded(decoder, document, idx=0):
        if "SENSITIVE_OUTCOMES" in document:
            raise RuntimeError("An opaque outcome subtree was decoded during header projection")
        return original(decoder, document, idx)

    monkeypatch.setattr(json.JSONDecoder, "raw_decode", guarded)
    generator = random.Random(340)
    for index in range(40):
        entries = list(
            {
                **expected,
                "report": {
                    "SENSITIVE_OUTCOMES": [None, 0, -2.5, False, {"quoted": 'escaped\\"{},[]:🙏'}],
                },
                "files": [{"content_base64": "opaque", "name": 'escaped\\"{},[]:🙏'}],
            }.items()
        )
        generator.shuffle(entries)
        payload = json.dumps(dict(entries), indent=2 if index % 2 else None, ensure_ascii=index % 3 == 0)
        assert json.loads(_source_header(payload)) == expected
    escaped_key = '{"\\u0070lan":{},"schema_version":1,"report":{"SENSITIVE_OUTCOMES":[0,null,-1]}}'
    assert json.loads(_source_header(escaped_key)) == {"plan": {}, "schema_version": 1}


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "[]",
        '{"report":[1}',
        '{"report":"unterminated',
        '{"plan" {}}',
        '{"plan":{}',
        '{"plan":{}} true',
        '{"plan":{},}',
        '{"plan":{} "report":{}}',
        '{"plan":{},"\\u0070lan":{}}',
        '{"schema_version":1,"unknown":0}',
        '{"plan":}',
    ],
)
def test_header_projection_rejects_invalid_framing(payload):
    with pytest.raises(ValueError):
        _source_header(payload)


def test_header_projection_enforces_utf8_and_structural_depth_bounds():
    with pytest.raises(ValueError, match="2 MiB"):
        _source_header('"' + "🙏" * 524288 + '"')
    with pytest.raises(ValueError, match="128"):
        _source_header('{"report":' + "[" * 129 + "0" + "]" * 129 + "}")
    assert _source_header(' {"report":' + "[" * 128 + "0" + "]" * 128 + "} ") == "{}"
