# Authored causal CI replay evidence, version 1

These small inputs and arithmetic are authored synthetic CI examples, not real
repository measurements or evidence of algorithmic superiority. All losses are
dimensionless; clocks are explicit UTC minute offsets. Seeds, calibration,
causal identification and independence are not claimed.

`timeline-oracles.json` is independently written before collecting SDK output.
It records expected decision/update counts, actions, loss sums and component
coverage. It is an oracle fixture format, not an SDK wire format.

The three continuation goldens contain the actual replay metadata and complete
state of the public host in `examples/replay/ci.py`. Every root has its explicit
schema version; `manifest.json` retains the file hash, continuation digest,
causal-prefix digest and independently checked arithmetic. These are portable
serialization examples of this bounded synthetic host, not certification of an
arbitrary consumer's state completeness or future schema compatibility.
Each human-readable JSON file ends with one newline. Its raw file SHA includes
that newline; the semantic continuation fingerprint uses canonical JSON without
file framing. The reader accepts the framed file without changing its object.

| Golden | Emitted decisions | Accepted rounds | Counts for a/b/c | Loss sums for a/b/c |
| --- | --- | --- | --- | --- |
| `chosen-continuation.json` | 3 | 1 | 1/0/0 | 1/0/0 |
| `component-continuation.json` | 3 | 1 | 1/1/0 | 0.5/0/0 |
| `full_panel-continuation.json` | 1 | 1 | 1/1/1 | 0/0.5/1 |

The chosen case retains two pending executions and an update eligible at 10:02
but applied at 10:03. The component case retains a partial root and a complete
successor: they produce one accepted round. The full panel retains realized,
modeled-counterfactual and synthetic origins separately for a, b and c, with
model/generator content bindings for the latter two. The numeric values are
authored; the origin distinction exercises the contract, not real-world truth.

Readers revalidate these v1 schemas, causal records, host statistics and bindings
without I/O or dynamic class imports. Restoring them requires explicitly selecting
the public `HostConsumerState` codec. No migration, executable pickle, physical
effect replay or durable acknowledgement is inferred. Unknown versions reject.
The original core, observation and objective fixtures are not rewritten.
`offline_identity_v1` explicitly simulates execution; it proves no physical CI
action. Process-death tests qualify a separately selected caller-owned aggregate
publication using the real immutable `ArtifactWriter`, not a mandatory engine
storage dependency or an exactly-once external-effect protocol.
