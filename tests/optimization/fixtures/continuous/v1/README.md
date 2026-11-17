# Continuous allocation v1 hand reference

These three files were authored from literal inputs and rational arithmetic,
using only the standard JSON/hash library. They were not produced by a solver
or by serializing a Coleman model.

For `Q=[[2,1],[1,2]]`, `c=(-5/2,-2)`, the unconstrained solution is `(1,1/2)`.
Under sum-one the optimum is `(3/4,1/4)`, with gradient `(-3/4,-3/4)`, equality
multiplier `3/4`, and objective `-25/16`. Clipping then renormalizing gives
`(2/3,1/3)`, worse by `1/144`. Under the different signed-box constraint
`a<=1/2`, the optimum is `(1/2,3/4)` with upper multiplier `3/4`; independent
coordinate clipping gives `(1/2,1/2)`, worse by `1/16`.

All coefficients and scores are dimensionless. The examples do not simulate
financial funding, borrow, wealth, physical costs or online observations. Wire
goldens preserve the strict full-law records, including explicit null lineage,
cash/net meanings, numerical limits and comparator. Future versions reject.
