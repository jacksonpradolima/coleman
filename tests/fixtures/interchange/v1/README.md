# Local JSON interchange v1 fixtures

These complete public synthetic fixtures exercise the optional outer codec. They
retain existing core-v1 and finite coverage-v1 contracts without migrating them.
No fixture is evidence that a built-in generic learner has shipped.

| Fixture pair | Meaning and independent expected facts |
| --- | --- |
| `coverage-*` | Three indivisible CI suites: A costs 2 s and covers x/y, B costs 2 s and covers y/z, C costs 1 s and covers z. Credits x/y/z are .5/.3/.2. A+C costs 3 s and covers credit 1 under budget 3 s; enumerating all eight subsets independently verifies the result. Production/receipt are explicitly supplied within the five-minute exchange deadline. |
| `learning_only-*` | A copied user-owned learner starts with tied cumulative dimensionless losses and selects recipe-a deterministically. Its state digest is an attestation of its exported typed state, without embedded executable state or an execution proof. |
| `full-panel-feedback-*` | A complete synthetic CI loss panel reports recipe-a = 1 and recipe-b = 0; both values use the same explicit dimensionless loss semantics and transform identity. The example's actual update gives cumulative losses (1, 0), so its next choice is recipe-b. |
| `mab_only-*` | A copied user-owned four-bin chosen-feedback learner uses all 625 exported Python MT state integers, samples recipe-a, and records its action-specific .75 mechanism mass. This is not an outcome probability or a cross-runtime RNG portability promise. |
| `feedback-*` | Only the sampled recipe-a row is delivered; delivered synthetic reward is zero, not missing. The actual copied host increments A's observation count without fabricating a B observation. |
| `unavailable-response.json` | The same input state remains unchanged and no numeric output or update is invented for unsupported work. |

`manifest.json` pins each exact file including its final newline and separately
records its typed canonical content fingerprint. These two digests intentionally
differ. The library codec revalidates versions, exact producing request/input,
implementation, state codec, result arithmetic and explicit deadlines. It does
not authenticate a producer or prove that a host actually performed an update.
The stdlib-only consumer example supports this finite fixture's declared route;
it is not a general replacement reader for every SDK contract or old Python SDK.
