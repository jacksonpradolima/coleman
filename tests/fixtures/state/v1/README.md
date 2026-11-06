# Authored CI state oracles

These fixtures contain public synthetic CI plans and arithmetic, with no measured
algorithmic advantage or physical CI execution claim. Each of 32 consumers owns
its state and random stream; operational isolation does not imply independent
scientific observations.

`ci-32-oracle.json` is an independent input/expected-value table. Child `i` emits
`D=2+i%3` opportunities and receives `U=1+i%2` chosen outcomes. Loss is
`((i+j)%4)/4`; integer quarter-loss sums avoid decimal ambiguity. Cadence advances
once per two releases. Plan A costs 3 seconds, B costs 5 seconds, and changing the
previous applied plan costs 2 additional seconds. The path oracle is
`3*D+2*count(B)+2*adjacent_changes`.

A preloaded least-significant-bit-first reservoir, word 4 with three remaining
bits, produces A,A,B without a backend refill. Each child's subsequent refill is
its own `random.Random(100+i).getrandbits(32)`. The listed bit vectors were
independently computed with CPython 3.14.6. They qualify this named sampling
profile; they are not a blanket guarantee about another API or Python version.
The complete backend words, index, Gaussian cache and buffered bits must survive
restart. Seeds and draw counts do not replace current state.

The pending checkpoint has D95/U48, 47 pending labels, loss16 and cadence42.
Virtual resource books sum to389 seconds; the actual interleaving of local round
then ascending child has25 B releases and8 switches, costing351 seconds. Those
are different paths. Completing pending labels gives U95/loss35.5 and changes no
RNG, decision, cadence or applied-path book.

The separate one-child long-run oracle has 4352 direct retained-replay decisions
with explicit one-round compaction and fresh-process JSON restoration at 2048
and 4096. No durable acknowledgement is claimed for those direct compactions.
Its actual final complete snapshot initializes the optional store; two further
real rounds and a separate durable seal qualify that handoff. The runtime host
retains sufficient counters, last plan, an eight-entry audit ring, bounded
cache/pending and complete RNG; it never retains a growing action history.

The five full wire fixtures listed in `../manifest.json` cover an actual
CPython MT19937 snapshot with a cached Gaussian value, a complete two-pending
host snapshot, and v2 pending, sealed and explicitly migrated continuations.
The migration definition fingerprint identifies the public test-owned mapping
in `tests/state/migrations.py`; it does not assert a framework codec history.

The manifest records raw file SHA-256 and canonical model fingerprints separately.
Each JSON file has exactly one final LF for source-file framing; that LF is
excluded from the canonical model fingerprint. These fixtures were materialized
from the frozen public example and CPython 3.14 state codec. They qualify exact
serialization and explicit reading. The separately authored literal counts,
losses, paths, RNG bits and cost tables provide the independent behavioral oracle.
A round-trip alone is not an independent correctness argument.

The manifest's `example_sha256` retains the original producer bytes. Its additive
`example_compatibility` metadata names a reviewed comment-only source variant;
none of the five golden payloads, file digests or model fingerprints was
regenerated. The `python-3.14-full-ast-no-locations-v1` profile parses Python 3.14
syntax and hashes the UTF-8 `ast.dump` output with field names and empty fields,
without source locations. Tests require both a registered source-byte digest and
equality with the original complete syntax-tree fingerprint. An unregistered
comment change does not automatically qualify, even when its syntax tree is
unchanged.

This limited qualification only preserves the interpretation of these existing
fixtures. It does not establish binary/runtime compatibility or authorize an
operational checkpoint resume. Changed example bytes still have different
source/code fingerprints, and the store's `ExecutionCompatibility` checks and
explicit known-mapping requirements remain in force.
