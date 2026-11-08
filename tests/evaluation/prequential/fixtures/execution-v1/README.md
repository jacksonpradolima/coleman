# Offline finite CI execution v1

`plan.json`, `context.json` and `step.json` capture complete canonical v1
exchanges from the actual shipped Hedge and causal replay. They check wire and
producer compatibility; captured wire is not an independent mathematical oracle.

`hand.json` is independently authored: the proposed full CI plan needs six
modeled seconds, the fast plan needs two and the declared limit is three.
Only fast is feasible. Coefficients remain the explicit `(1/2, 1/4, 1/4)` prior
before any feedback; one decision and one domain call produce zero updates.
These are modeled inputs, not measured process durations or empirical CI data.

The context binds the earlier complete coefficient record and prior domain
state. Later outcomes, model charges and source-access evidence are separate
records. An action receipt is neither an accepted checkpoint nor physical
execution proof. Unknown versions reject; existing result readers are unchanged.
