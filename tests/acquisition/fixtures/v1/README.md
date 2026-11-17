# Terminal allocation and fidelity evidence

`terminal-plan.json` is the original independently authored terminal-only plan.
It remains unchanged and readable. Its signed objective and Decimal resource
meaning are independent of early-elimination qualification.

`qualified-fidelities.json` is independently authored qualification input. Its
declared targets are 1, 2 and 4 build replays; early scores are 1/4 and 1/2, with
terminal score 1/2. The absolute discrepancies are 1/4 and 0, each within the
declared tolerance 1/2. These values exercise the contract; this constructed case
provides no held-out or population-error guarantee. Tests compute its canonical
JSON/digest independently with the standard library.

`prototype-unbound-qualification.json` is an actual unsupported pre-integration
writer artifact from commit `1d5b4a58d4304813966cd57475bbf7de82d490d7`, using
noneditable wheel SHA-256
`7549c95894481119da528dff2da686f254587c6d8dc610de7aa2fcffa756f958`.
Its recorded file SHA-256 is
`4aa101d3dc31a2952dcb5aba7be094f708c2c0b1d176427c9a189ec880806ea0`.
The old writer retained two early scores without identifying their resource
targets. Its reader incorrectly admitted seven same-count changes of targets,
versions, identities or units while retaining the same evidence fingerprint.
The corrected reader explicitly rejects this incomplete artifact. It must never
infer the missing schedule from a caller's later plan. Original training evidence
is needed to reconstruct a bound qualification; an alternative is a new
terminal-only experiment.
