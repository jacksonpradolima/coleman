# Learning wire fixtures

The reference configuration and rational traces are explicitly authored public
CI examples. The controlled law and two complete snapshots are SDK-produced
version-one compatibility fixtures, reviewed with the hand calculation below.
They are not independent numerical solver oracles; those live in
`tests/optimization/_independent_oracles.py`. No test refreshes these files.

The controlled fixture declares three CI experts, bounded dimensionless losses,
initial coefficients `(1/2, 1/4, 1/4)`, static boxes `[1/8, 3/4]`, a researcher
target `(1/4, 1/2, 1/4)` with share one, and a full-L1 strength `1/16` in
coefficient units. The learning rate is zero, so exponential arithmetic adds no
platform-dependent transcendental result to the fixture. Two observed panels
form each release block. A separately declared movement period allows `1/2`
absolute variation per coordinate; these coefficients do not represent trades
or resource charges.

At the first release the proposal is exactly the target. With multiplier zero,
soft thresholding the differences `(-1/4, 1/4, 0)` at `1/16` gives the unique
quadratic-plus-L1 optimum `(5/16, 7/16, 1/4)`. The exact accumulated movement is
`(3/16, 3/16, 0)`. The next proposal is again the target; its differences from
the emitted vector are `(-1/16, 1/16, 0)`, so the second nonempty release is an
optimal hold. Its release counter advances while its exact movement stays
unchanged. All constants are dyadic, including the atom counts on the
`2^-1074` coefficient grid.

`controlled-pending-state-v1.json` stops after the first accepted panel (`U=1`,
`R=0`). `controlled-released-state-v1.json` stops after the moving and holding
releases (`U=4`, `R=2`). The latter retains the complete pre-attempt exact ledger
in its latest receipt, separately from the current ledger. Its law binding
stays fixed as dynamic origin versions change. The tests read through the
bounded explicit-payload codec, restore actual learners, continue the pending
snapshot to the released bytes, and reject unknown versions and fields.

Clock values are consciously authored UTC mechanics, not actual CI jobs or
proof of historical source access. Source/transform declarations remain caller
attestations, not authentication. Neither these fixtures nor the comparison
streams establish algorithm superiority, independent empirical observations,
calibrated probabilities or financial applicability.
