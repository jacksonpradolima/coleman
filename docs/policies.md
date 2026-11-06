# Choose policies and rewards

A policy determines how accumulated feedback becomes a ranking. A reward defines
the learning signal. Keep the source, feedback visibility and budget fixed when
comparing policies; [comparable baseline matrices](comparable-baselines.md)
provide an explicit candidate-grid workflow.

## Start with simple comparators

Use `Random` as an uninformed ranking comparator and `UCB` as a small adaptive
starting point. Add history-based heuristics through the baseline matrix before
interpreting the value of a more elaborate method. The
[original COLEMAN study](https://doi.org/10.1109/TSE.2020.2992428)
provides the research context for bandit-based TCPCI.

## Available native policies

The native runner recognizes these **37 names**. Required constructor parameters
belong in the corresponding reward branch of `algorithm`; some windowed methods
also require `window_sizes`. Use the [configuration reference](configuration.md)
and [constructor API](api/policy.md) for the exact fields. A configuration pack
is a preset for a subset of these names, not the complete policy inventory.
The [native decision walkthrough](native-policy-semantics.md) explains the actual
37 mechanisms, their factory dispositions and an explicit pre-evaluation record
available in the unreleased v5 integration source. Three named windowed factory
routes need explicit refusal rather than being advertised as successful recipes.

| Family | Native configuration names | What changes in the ranking |
| --- | --- | --- |
| Random and greedy | `Random`, `Greedy`, `EpsilonGreedy`, `DecayEpsilonGreedy`, `OptimisticGreedy`, `EpsilonDecreasing` | Start with random order, exploit accumulated value, or vary explicit exploration |
| Confidence-based | `UCB`, `UCB1`, `UCB2`, `KLUCB`, `UCBTuned`, `UCBV`, `MOSSUCB` | Combine accumulated value with different exploration/confidence constructions |
| Bayesian and stochastic | `ThompsonSampling`, `BayesianUCB`, `BootstrappedThompson`, `Softmax`, `Pursuit`, `PHE` | Rank sampled scores, posterior scores or maintained preferences; Softmax and Pursuit sort preferences without sampling them |
| Adversarial-inspired ranking | `EXP3`, `EXP3IX` | Sort exponential-weight scores and update all current names using native reward deltas |
| Changing-history | `SlidingWindowUCB`, `DiscountedUCB`, `ChangeDetectionUCB`, `FRRMAB`, `SlMAB` | Adapt the use of older feedback through windows, discounting, detected changes or rank-based credit |
| Contextual | `LinUCB`, `SWLinUCB`, `LinTS`, `SWLinTS`, `ContextualEpsilonGreedy`, `SWContextualEpsilonGreedy` | Condition a test's score on configured feature vectors |
| Combinatorial-inspired ranking | `CombinatorialUCB`, `CombinatorialThompson` | Return the complete score order; the native `ordered[:k] + ordered[k:]` operation selects no subset |
| Dueling-inspired ranking | `DuelingUCB`, `PairwiseThompsonRanking` | Rank pair estimates updated from the policy's own previous order, without observed pairwise preferences |
| Policy portfolio | `PortfolioUCB` | Select candidate ranking policies sharing one underlying agent and its native values |

The policy families expose different hypotheses to test. Keep their actual native
reward/update and feedback contracts visible: sharing an algorithm name does not
establish the regret guarantee or feedback model of the original publication.
The actual factory paths for `SlMAB`, `SWLinTS` and
`SWContextualEpsilonGreedy` select agents missing the required window capability.
`SlidingWindowUCB` takes a mean-ranking fallback and ignores factory windows.
The [catalog and migration guidance](native-policy-semantics.md#the-37-native-ranking-mechanisms)
distinguish these paths from separately constructed window agents. Existing raw
behavior is preserved; the explicit capture API refuses incompatible routes.

## Connect a policy to its research

| Research foundation | What it helps you understand | Start with |
| --- | --- | --- |
| Auer, Cesa-Bianchi and Fischer, [*Finite-time Analysis of the Multiarmed Bandit Problem*](https://doi.org/10.1023/A:1013689704352), 2002 | Classical exploration through confidence bounds under the paper's reward assumptions | `UCB1`, then compare the configurable UCB variants |
| Li, Chu, Langford and Schapire, [*A Contextual-Bandit Approach to Personalized News Article Recommendation*](https://arxiv.org/abs/1003.0146), 2010 | Linear contextual exploration | `LinUCB`, with features actually available before the CI decision |
| Chen, Wang and Yuan, [*Combinatorial Multi-Armed Bandit: General Framework and Applications*](https://proceedings.mlr.press/v28/chen13a.html), 2013 | How subset actions, their observed components and reward structure affect a bandit problem | Combinatorial policy interfaces and their actual subset/full-ranking behavior |
| Prado Lima and Vergilio, [*A Multi-Armed Bandit Approach for Test Case Prioritization in Continuous Integration Environments*](https://doi.org/10.1109/TSE.2020.2992428), 2022 | Applying adaptive feedback to volatile CI test suites | Native policies, RNFail/TimeRank, scheduling budgets and repeated builds |

For complete citations and the HCS/dataset/evaluation studies, follow the
[research guide](research.md). Evaluate extensions under the actual CI workload,
source availability, feedback and cost protocol.

## Compare exploration settings

The [README sweep example](https://github.com/jacksonpradolima/coleman#sweep-engine)
runs three UCB exploration coefficients across two seeds using the existing sweep
engine. Compare configuration choices on a declared validation period, retain
simple baselines, and freeze the evaluation protocol before protected outcomes.
[Comparable baseline matrices](comparable-baselines.md) bind the candidate grid,
observations and execution costs; [protected evaluation](protected-evaluation.md)
provides the separate inference path.

## Rewards and quality metrics serve different purposes

`RNFail` and `TimeRank` are native reward choices. RNFail uses failure feedback;
TimeRank additionally incorporates ranking information. Other reward classes
are described in the [reward API](api/reward.md). Check the runner's supported
configuration names before using a class name as a config value.

A reward transforms feedback for learning. NAPFD and APFDc describe prioritization
quality under their own definitions. Neither a reward nor an internal score is a
calibrated failure probability or a statistical comparison.

See [native result interpretation](evaluation-guide.md#read-native-results) and
[versioned measurements](domain-results.md) for the corresponding output contracts.

## Contextual policies

Contextual policies consume configured feature columns, with previous-build
features declared under `contextual_information.config.previous_build`. The
current execution's eventual duration and verdict must not leak into the
pre-decision feature vector. A previous-row transformation alone does not prove
source availability; retained timestamps and source meaning still matter.

[Li et al.'s contextual-bandit paper](https://arxiv.org/abs/1003.0146) introduces
LinUCB in a recommendation setting. Its presence here motivates a policy family;
CI effectiveness needs its own source-appropriate evaluation.

Continue with [point-in-time observations](point-in-time-observations.md),
[protected evaluation](protected-evaluation.md) and the
[research reading path](research.md).

## Extend a policy

Use the native policy/agent interfaces and lifecycle hooks described in
[extensions and parallelism](extensibility.md). Reuse orchestration, sinks and
checkpoints. Document update state, randomness, feedback assumptions and the
conditions under which a score is meaningful.
