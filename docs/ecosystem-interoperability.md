# Connect an external tool without losing experiment meaning

COLEMAN can supply the typed decision, recovery and evidence boundary around an
external algorithm. That does not make every external API interchangeable with
a COLEMAN learner, or make an upstream checkpoint safe to resume.

**Assessment:** retain the existing implementations; select no external adapter
yet. The public CI tuning example below is a concrete comparison target, but it
does not establish a need that its existing sweep cannot meet. An adaptive
search, forecasting, solver or environment bridge needs its own declared public
experiment and maintenance case before implementation. This is a feasibility
assessment, not a shipped integration or a v5.0 release requirement.

## Start with the boundary that already exists

This assessment inspected integration source
[`6c72ca3`](https://github.com/jacksonpradolima/coleman/tree/6c72ca343efba345a1d540e2473227f047093b58).
The architecture, typed contracts, lightweight installation and frozen evaluation
prerequisites are integrated. Their completed delivery does not qualify an
external package.

| Existing boundary | Reuse it for | What it does not establish |
| --- | --- | --- |
| [`Decider`, `Updater`, `Stateful`, `Solver`](https://github.com/jacksonpradolima/coleman/blob/6c72ca343efba345a1d540e2473227f047093b58/coleman/core/protocols.py) | Narrow direct calls without a workspace or framework superclass | Correct mathematics, persistence or idempotent updates |
| [Actions and capability negotiation](decision-contracts.md) | Explicit choice, subset, slate, mixture, vector or acquisition; feedback coverage, units, timing and state ownership | That similarly shaped arrays have the same scientific meaning |
| [Sampling records](discrete-feedback.md) | Bind actual finite selection mass to the executed action | A ranking score, mixture coefficient or continuous density being a propensity |
| [Owned state and recovery](state-recovery.md) | Complete selected codecs, owned RNG, actual acknowledged progress | Automatic conversion of opaque external state or exactly-once external effects |
| [Typed durable tasks](typed-execution.md) | Approved implementation capture, existing workers, physical attempts, bounds, artifacts and reconciliation | A second study engine or scheduler being necessary |
| [Frozen evaluation](evaluation-protocol.md) | Development selection, source exposure, actual clocks, complete panels and independent units | That tuning trials or seeds are independent scientific observations |
| [Native sweeps](api/spec-sweep.md) | Deterministic grid/zip/seed expansion of existing `RunSpec` inputs | Outcome-driven search or generic external optimizer orchestration |

The minimal installation requires Pydantic and packaging. No package in this
assessment is added to that dependency closure, discovered at import time or
selected by executable request text. Existing TCPCI results, readers, policies
and sweep behavior retain their meanings.

## Versions and packaging evidence

The following are exact upstream release artifacts inspected on **18 November
2026 UTC**. Metadata and source inspection are not clean-install, performance,
security or continuation qualification. A wheel filename is availability
evidence, not an execution result. Licenses below concern the named package;
dependencies, datasets, environments and model artifacts require their own terms.
The [primary-source receipt](research/ecosystem-interoperability-sources.json)
records exact release metadata and selected source-member hashes, including the
dependency declarations missing from OBP's published metadata.

| Candidate | Inspected version and package license | Declared runtime and dependency cost | Packaging observation |
| --- | --- | --- | --- |
| Vowpal Wabbit | [`vowpalwabbit 9.11.9`](https://pypi.org/project/vowpalwabbit/9.11.9/), BSD-3-Clause | Python ≥3.10; native extension, with no Python `Requires-Dist` entries declared | CPython 3.14 wheels include macOS arm64/x86-64, Linux and Windows; this does not qualify a particular ABI/image |
| River | [`river 0.27.0`](https://pypi.org/project/river/0.27.0/), BSD-3-Clause | Python ≥3.11; NumPy, SciPy and Narwhals; pandas is optional | CPython 3.14 wheels are listed, including macOS universal2 and Linux/Windows |
| Optuna | [`optuna 5.1.0`](https://pypi.org/project/optuna/5.1.0/), MIT | Python ≥3.10; NumPy, SQLAlchemy, Alembic, packaging, PyYAML, colorlog and tqdm | Universal Python wheel; a database and sampler add operational responsibilities |
| Open Bandit Pipeline | [`obp 0.5.7`](https://pypi.org/project/obp/0.5.7/), Apache-2.0 | Published metadata leaves Python/dependency requirements incomplete; released `setup.py` declares a scientific/plotting stack and Torch | Source distribution only; classifiers list Python 3.8–3.10. Python 3.14 compatibility is **unqualified**, not disproved by classifiers alone |
| Gymnasium | [`gymnasium 1.4.0`](https://pypi.org/project/gymnasium/1.4.0/), MIT | Python ≥3.10; NumPy, cloudpickle, typing-extensions and Farama notifications; environment extras add separate dependencies | Universal base wheel; base availability does not qualify physics engines, ROMs or rendering |
| OSQP | [`osqp 1.1.3`](https://pypi.org/project/osqp/1.1.3/), Apache-2.0 | Python ≥3.8; NumPy, SciPy, Jinja2, setuptools, joblib and a native solver | CPython 3.14 wheels include macOS arm64/x86-64, Linux and Windows |

These releases were not installed for this assessment. Dependency counts above
are direct declarations, not a measured transitive closure, installed size,
startup latency or vulnerability verdict. In particular, absent OBP wheel
dependency metadata must not be reported as an empty dependency closure.

OSQP is the convex optimizer considered here. CVXPY is a useful modeling layer,
but adding it would introduce a further abstraction and solver selection before
this assessment has identified a problem requiring either. [CVXPY describes
that distinction explicitly](https://www.cvxpy.org/).

## Map operations before choosing an adapter

| Ecosystem operation | Plausible COLEMAN boundary | Required retained evidence | Mapping that remains unsupported |
| --- | --- | --- | --- |
| VW `Workspace.predict` / `Workspace.learn` | A separately qualified `Decider`/`Updater` profile with choice actions and chosen feedback | Original context/action order, actual executed member, exploration mechanism, chosen propensity, cost transform and full external state | Calling native UCB1/Exp3 contextual; treating returned scores as exact sampled-action mass |
| River `predict_one` / `learn_one` | A prediction/learning adapter with typed features, target units and original producer joins | Entire pipeline, preprocessing, optimizer, sufficient statistics, RNG and pending original inputs | Calling a forecast calibrated probability, or assuming every model shares one portable checkpoint |
| Optuna `Study.ask` / `Study.tell` | A development-only configuration selector outside the existing task executor | Study/trial identity, frozen search law and space, sampler state, proposal order, complete outcomes and every physical attempt | Treating a trial as a scientific unit, an action propensity, or an automatically idempotent task |
| OBP policy evaluation | A separately qualified evaluation capability over genuine logged choice data | Original action/position, behavior propensity, target policy probabilities, reward, source eligibility and support assumptions | Replacing full-information or stateful evaluation by relabeling arrays; claiming an implemented OPE capability |
| Gymnasium `reset` / `step` | A specific stateful problem/execution adapter | Original observation/action/reward, terminated and truncated separately, environment/wrapper state and clocks | Generic environment checkpointing, stateless chosen-action replay, or external-effect reconciliation |
| OSQP `setup` / `solve` | A solver-only adapter with its own result and numerical profile | Ordered variables, objective/constraints/units, solver/backend/settings, status, residuals and measured work | Reporting OSQP output as the native active-face solver profile or accepting “inaccurate” as qualified optimality |

These are formulation mappings. No row declares an installed adapter, new
capability or inherited upstream guarantee.

The maintenance comparison is qualitative, not a benchmark score:

| Candidate | Additional surface a maintainer would own | Current reason to retain the baseline |
| --- | --- | --- |
| VW | Feature/index translation, exploration, native ABI and complete model/RNG continuation | A contextual public experiment has not been selected; native fixed-arm methods retain their narrower qualified contract |
| River | A selected estimator's entire preprocessing/optimizer/drift pipeline and serialization policy | The existing append-only ridge serves the qualified forecast examples; a different forecast need remains to be specified |
| Optuna | Study/sampler/pruner versions, proposal order and the join between study history and actual task attempts | The public three-value CI sweep is already finite and reproducible |
| OBP | Logged support/position mappings, estimator assumptions and a source-only legacy dependency stack | Off-policy evaluation remains an explicit research gate, not a missing wrapper around existing inference |
| Gymnasium | A particular environment's full hidden state, wrappers, effects and interruption policy | An environment seed does not supply the state/effect contract |
| OSQP | Native backend, sparse representation, tolerances, statuses and a separate result codec | The existing small SPD solver covers the current qualified coefficient examples |

These surfaces could be worth maintaining. Their presence is not an estimate of
developer hours, paid costs or a claim that another package is inferior.

### Vowpal Wabbit: preserve the sampling mechanism

The [official contextual-bandit tutorial](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html)
describes a context, an actual chosen action and feedback only for that action.
A bridge must freeze the translation between versioned COLEMAN members and
upstream action indexes/features, including any exploration and reward-to-cost
conversion. Post-filtering or reordering an action distribution changes that
contract.

An upstream floating distribution is not proof of its sampler's effective
finite mass. Declare exact evidence only for the actual qualified mechanism;
otherwise retain qualified approximation or unavailable evidence. The shipped
COLEMAN discrete profiles have fixed catalogs, immediate untimed feedback and
exact sampling requirements: one through 256 arms, one pending choice, and
dimensionless observed rewards in `[0, 1]`. They are not wrappers for arbitrary
contextual VW policies. Model persistence, pending decisions and complete exploration/RNG state
need an independent restart fixture before durable continuation can be claimed.
The [Python API](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/reference/vowpalwabbit.pyvw.html)
must be selected with its exact options: a training label's “cost” is not evidence
that money or a physical resource was paid.

### River: own the complete learning pipeline

River's [online prediction/update interface](https://riverml.xyz/latest/api/overview/)
is a promising direct-use shape. Its algorithms and preprocessing still need
individual contracts: original features used to predict must accompany the
later label, units and missingness cannot be inferred from dictionary keys, and
pipeline or optimizer state must not disappear during recovery. Any prediction
side effects belong in the declared state transition.

COLEMAN already supplies an independently usable [append-only ridge
reference](continuous-decisions.md). Another regressor is justified by a public
need that reference cannot meet, not by a larger model catalog. Neither generic
Python serialization nor replaying only the latest weights establishes portable
cross-version recovery. An initial external profile may explicitly support only
fresh direct execution until its entire state is qualified.

### Optuna: distinguish study history from a recoverable experiment

The [ask-and-tell interface](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html)
lets a host separate proposal from execution. A potential bridge would let
Optuna propose a declared configuration and let COLEMAN's existing executor
perform work and retain evidence. It should not add another worker pool.

The current [task plan](https://github.com/jacksonpradolima/coleman/blob/6c72ca343efba345a1d540e2473227f047093b58/coleman/tasks/contracts.py)
freezes its complete coordinate set before execution. It does not append new
trials after each result. A future outer study must bind each proposed immutable
task to its original trial and enforce a study-wide budget and exposure history;
creating another plan must not reset those limits. That composition is not an
implemented adaptive-study route.

Database trial history and sampler continuation are different state. The
[upstream persistence guidance](https://optuna.readthedocs.io/en/stable/faq.html)
does not promise cross-version pickle recovery; the
[RDB tutorial](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html)
separately restores a seeded sampler. The
[RDB storage API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html)
also states that its heartbeat mechanism is for `optimize`, not `ask`/`tell`.
Do not infer missing-worker recovery from choosing a database.

After an uncertain response, reconcile the original trial and immutable outcome
before another `ask` or `tell`. An option that skips already-finished trials does
not itself prove that a retried value matches the original. Preserve failures,
pruning, sampler RNG and completion ordering. Parallel scheduling can change
which observations influence the next proposal even with the same seed.

### Open Bandit Pipeline: preserve estimator assumptions

The [OBP documentation](https://zr-obp.readthedocs.io/en/latest/) describes logged
feedback and behavior propensities for off-policy evaluation. The relevant
COLEMAN research boundary remains [optional off-policy estimation](https://github.com/jacksonpradolima/coleman/issues/377).
An available estimator does not satisfy that entry gate.

A proposed export must distinguish chosen feedback from modeled alternatives,
bind action positions and policy probabilities, and refuse absent or zero
support required by its selected estimator. Continuous densities, inclusion
probabilities and whole-slate probabilities cannot be substituted for each
other. A context/action/reward table also does not establish independence or
stateful counterfactual validity. Code licensing does not admit the Open Bandit
Dataset or any other source under a new scientific scope.

### Gymnasium: an environment seed is not a checkpoint

The [environment API](https://gymnasium.farama.org/api/env/) separates `step`
from `reset` and returns both `terminated` and `truncated`. Preserve that
distinction, including its effect on any selected learning target. Observations
may be partial; `info` is not a declaration of complete recoverable state.

A specific environment requires a versioned observation/action mapping and
complete environment, wrapper, RNG and pending-effect state. Resetting with the
same seed is a new rollout, not continuation of an interrupted one. Replay or
retry must not repeat an unacknowledged physical action. The
[stateful-control research boundary](https://github.com/jacksonpradolima/coleman/issues/371)
therefore remains distinct from the synchronous chosen-action learners.

### OSQP: retain a separate numerical result profile

OSQP accepts convex quadratic objectives and linear constraints through
[`setup(P, q, A, l, u)` and `solve`](https://osqp.org/docs/interfaces/python.html).
COLEMAN's current `ContinuousProblem` is a smaller contract: one to four
dimensionless coordinates, supported box/simplex constraints and a strictly
positive-definite quadratic objective. Its result profile names the native
rational active-face algorithm. An external solution must not claim that
algorithm's identity, certificate or numerical envelope.

For a later independent parity fixture, take `Q = I`, `q = (-3/4, -1/4)`,
`x >= 0`, `sum(x) = 1`. The hand solution is `x = (3/4, 1/4)`, objective
`-5/16`, with zero gradient and equality multiplier zero. This is worked
arithmetic, not an executed upstream test. A real bridge must separately test
active bounds, infeasibility, units, finite conversion and tolerance failures.

Retain every [upstream termination status](https://osqp.org/docs/interfaces/status_values.html).
“Solved inaccurate,” iteration/time limits and an available vector cannot be
silently converted into accepted optimality. Bounds represented upstream by
infinities need an explicit adapter convention; nonfinite numeric observations
remain forbidden. Cold re-solving a frozen problem can be an explicit initial
profile; warm-start vectors are not proof of a portable internal factorization
checkpoint. Neither route executes a domain action or turns a regularizer into
a paid cost.

## A concrete public CI comparison target

The existing [CI policy guide](policies.md) and
[`test_sweep.py`](https://github.com/jacksonpradolima/coleman/blob/6c72ca343efba345a1d540e2473227f047093b58/tests/spec/test_sweep.py)
already exercise UCB exploration coefficients. Use that workflow to formulate a
need before choosing an adaptive search tool.

| Stage | Existing baseline and evidence required |
| --- | --- |
| Input | Declared development CI source, native policy/metric semantics, `c` values `(0.1, 0.3, 0.5)` and two declared seed coordinates |
| Proposal | Existing grid expansion produces six concrete run specifications; seeds remain repeat coordinates, not six independent repositories |
| Execution | Existing native execution produces actual result/artifact records; retain failed attempts and measured seconds/bytes separately from logical trials |
| Selection | Predeclared development objective and complete aggregation rule; unavailable cells do not become zero or disappear |
| Confirmation | Freeze selected configuration before protected access; use actual independent units and the existing evaluation boundary |
| Output | Selected configuration or retain-baseline disposition, complete attempt/cost evidence and supported uncertainty status |

This table formulates a comparison; this assessment has not executed those six
runs. A fixed grid of three coefficients is not evidence that a Bayesian search
engine is needed. The public source does not yet provide a selected larger or
conditional search experiment with frozen budgets and a maintenance comparison.

If that need is supplied, assess **at most one initial Optuna ask/tell bridge**
against the existing grid and a declared simple random-search baseline. This
identifies a first candidate for evaluation, not a selected adapter or an
implementation commitment. Compare the same source, development access,
configuration work and maximum budget; report actual spent work and unused
budget. Match physical attempts, startup/setup, CPU time, memory, artifact bytes
and any declared paid charges. An optimizer's trial count alone is not a cost
comparison or evidence of better search.

### What the research motivates

[OpenTuner (PACT 2014)](https://commit.csail.mit.edu/papers/2014/ansel-pact14-opentuner.pdf)
motivates separating a domain's configuration representation, measurement and
search technique. It does not establish that COLEMAN should rebuild its search
ensemble or inherit its reported improvements.
[HPOBench (NeurIPS Datasets and Benchmarks 2021)](https://arxiv.org/abs/2109.06716v3)
provides a route to reproducible multi-fidelity experiments, including distinct
containerized, surrogate and tabular problems. Each chosen artifact still needs
version, terms, fidelity, disclosure and physical-work admission. COLEMAN's
existing [finite acquisition baseline](subset-acquisition.md) must remain a
comparison candidate where its assumptions fit.

[Contextual bandits with packing and covering constraints (JMLR 2024)](https://jmlr.org/papers/v25/24-1220.html)
motivates explicit resource vectors and constrained comparators. Combining an
arbitrary regressor and optimizer does not implement that method or transfer its
guarantees. The [OSQP paper (Mathematical Programming Computation 2020)](https://web.stanford.edu/~boyd/papers/osqp.html)
motivates the separate convex-solver route, with its own algorithm, numerical
checks and assumptions. None of these sources supplies COLEMAN efficacy or
replaces the experiment entry decision.

## What would reopen implementation

Record a concrete public experiment containing the missing need, exact action
and outcome meanings, candidate/baseline laws, units, source/availability clocks,
development-only search space, finite work/cost limits and expected maintenance
benefit. Review that formulation before adding an adapter. A constructed public
experiment is allowed; neither a new historical dataset nor an adaptive win is
a blanket prerequisite.

An accepted first bridge would then need direct-upstream parity on independent
fixtures, conflicting-retry rejection, full restart or explicit unsupported
continuation, real interrupted/concurrent execution where applicable, and
noneditable installed tests outside the checkout. Capture exact package,
dependency, backend and source versions. Keep a clean core-only environment
separate from the selected optional environment, and prove cold-import isolation.
Reuse existing state, tasks and evaluation rather than rebuilding them.

For now the disposition is **feasibility assessment completed; adapter selection
and implementation deferred pending entry evidence**. None of VW, River,
Optuna, OBP, Gymnasium or OSQP is advertised as an implemented COLEMAN capability.
The native baselines remain available, and no release publication follows from
this assessment.
