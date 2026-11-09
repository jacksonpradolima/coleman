# An installed external Hedge executor

This public fixture wraps the shipped constant-rate `Hedge`; it introduces no
new learner or efficacy claim. It connects one independently approved installed
provider to the common task coordinator and the existing `LocalStateStore`.
Ordinary `Hedge` construction and `observe` calls remain independent of the task
service, executor recipe, SQLite and provider registration.

The three experts start at `(1/2, 1/4, 1/4)`, with `eta = log(2)`. Each original
panel is fully observed, synthetic normalized loss with explicit unit `1`, role
`loss`, bounds `[0, 1]` and a pinned identity transform:

| Panel | Expert A | Expert B | Expert C |
| --- | ---: | ---: | ---: |
| 1 | 0 | 1 | 1 |
| 2 | 1 | 0 | 0 |
| 3 | 0.25 | 0.75 | 0.5 |

Zero is observed loss; unavailable/null outcomes refuse. No RNG is invented.
The helper distribution actually selects the original next panel and is an
active transitive dependency of the provider, so its installed bytes belong to
the captured implementation closure too.

## Install and select the public fixture

Use an ordinary noneditable base COLEMAN wheel, then build and install the two
ordinary fixture distributions from `support/` and `provider/` with the same
consumer interpreter. The host's installed qualification builds these wheels;
the provider never installs or resolves dependencies itself. See the repository
[installation guide](../../docs/core-installation.md) for the base environment.

Copy `ci.py` and the shipped `coleman_typed_hedge/recipe.json` resource outside
the checkout. The static recipe is complete JSON and can be inspected without
importing the provider. Its recipe/schema fingerprints declare meaning; they
do not approve or authenticate code. An operator deliberately captures and
approves the actual supported installed image:

```console
/path/to/consumer/bin/python -I -B /outside/ci.py approve /outside/recipe.json /outside/approved.json
/path/to/consumer/bin/python -I -B /outside/ci.py run /outside/workspace /outside/approved.json
```

Use a stdlib `venv` on a trusted local POSIX filesystem with kernel locks and
no-follow directory operations. The initial profile requires CPython with
optimization zero, a standard import context and immutable directory-wheel
installations. `-B` prevents new bytecode-cache writes; existing
installed files are captured rather than silently ignored. Capture checks the
selected entry point, resource ownership and effective installed dependency
closure before loading the provider. It does not attest undeclared dynamic
imports, native libraries, the host filesystem or code already changed in memory.

The approval object is supplied explicitly to `TaskService`. Workspace metadata,
an imported plan or a saved matching profile is an equality constraint, never
permission to import a package. An already running fixed supervisor retains its
trusted bootstrap context after client disconnect. After that daemon dies, the
operator supplies the profile again. Task requests contain concrete bounded
values and a registered recipe reference; they accept no import path or loader.

## Follow the actual state boundary

`sample.configuration()`, `inputs()` and `planning_request()` expose the complete
typed data. Input construction briefly creates a direct learner to export its
actual immutable virgin snapshot; it retains no live owner. `recipe()` performs
no capture, source access, owner construction or store I/O. The provider's five
value codecs, complete state codec and progress-proof codec are declared
separately, with explicit UTF-8 byte capacities and strict v1 readers.

Preparation checks the complete source binding and configuration. Each selected
trajectory must match the original input bytes, exact reference/version, work
profile and terminal panel count before state access or restore. The optional
durable layer creates fresh actual owners only inside accepted store staging.
It serializes the **complete** inner `AlgorithmState[ReferencePayload]` inside a
separate `HedgeCheckpoint` envelope carrying the source, plan, coordinate and
captured implementation. Runtime bindings never alter the mathematical state.

Two accepted original panels are committed in the first transition and one in
the second. Consequently the terminal logical frontier is **U3**, while the
storage generation is **2** after generation-zero initialization. Neither
generation nor physical invocation count substitutes for a learner update count.
The example compares complete inner state bytes against an independent direct
run of the same shipped learner and original panels, including sufficient
statistics, compensation, weights, fixed configuration and mathematical owner.

The progress callback runs after a real ACK. If its response is lost, recovery
opens the actual current collection, reads every complete state field and
reissues the exact original command from the retained manifest. The store returns
its real retained receipt before factories, restores or transition callbacks.
The verifier joins that receipt and current head to decoded U; it does not
fabricate acceptance from a JSON file or a requested work integer. A final
coordinate result retains this proof, and the common driver owns complete-last
job publication. The stage boundary invokes the common loader once and reports
`constructed_mechanism_only`; it supplies no selection, inference or promotion.

An immutable definition marker makes a missing previous state directory refuse
instead of silently becoming a new learner. Corruption and incompatible
configurations also refuse. Whole-subtree deletion or rollback together with all
independent evidence requires an externally retained trusted minimum to detect;
this fixture adds no remote effect protocol or hostile-filesystem sandbox.
Trusted factories and restore bodies must stay bounded and free of external
effects. Direct and prepared owner calls are sequential; outer ownership/CAS
provides the supported cooperative concurrency boundary, not locks in the math.
