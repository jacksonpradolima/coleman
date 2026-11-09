# Issue #365 — cloud handoff

**Status: unfinished WIP, locally committed for transfer. Not ready for push, PR,
merge, issue closure or release.** The user paused implementation and then
requested this handoff and local commit. This handoff does not authorize
publication or starting another issue.

**Execution agreement:** [AGENTS.md](AGENTS.md) is the persistent local/cloud
working agreement: issue order, scoped agent ownership, actual independent
review, pinned local-only OpenCodeReview commands/rules, frozen validation,
complete file coverage, consolidated pushes, PR feedback and protected merge.
Read it before the issue-specific state below. This documentation follow-up
does not restart the paused implementation or qualify the WIP for publication.

## Repository and next task

- Public repository: <https://github.com/jacksonpradolima/coleman>.
- Task: [#365 — continuous allocation and independent online forecasting](https://github.com/jacksonpradolima/coleman/issues/365).
- Parent: [#353](https://github.com/jacksonpradolima/coleman/issues/353).
- Public roadmap: <https://github.com/users/jacksonpradolima/projects/3>.
- Working branch: `feat/v5-365-continuous-prediction`.
- Integration/PR target: **`release/v5`**, never `main` for this issue.
- Starting integration commit: `9a1150f051773c9e73f6b95a68c9471f4021a052`
  (merged TCPCI PR #417 / completed issue #369).
- Local main at handoff: `1edbee874129aa85e26e592d9649826e85e1ac6b`.
- Package metadata remains **4.20.0**. One stable v5 publication belongs to #381;
  integration issue branches must not publish intermediate releases.

Read applicable `AGENTS.md`, the live issue and **all comments**, parent and
relevant dependencies before continuing. No applicable `AGENTS.md` was present
in the local repository or exact ancestor paths when implementation started.
The handoff follow-up now adds the root working agreement. #365 had no comments
when first read. Its prerequisites #355, #357, #361 and #360 were completed.
Public Project 3 was updated from stale Blocked/Todo to Ready/In Progress; the
issue remains open and its acceptance checklist is not marked complete.

Use only public COLEMAN sources, public Project 3 and generic authored examples.
Do not search nonpublic consumers, projects, attachments or sibling repositories.
Do not copy private names, examples, schemas, datasets or decisions into public
work. A design or illustrative issue API is not implemented evidence.

## What is already in this branch

| Area | Implementation and boundary |
| --- | --- |
| `coleman/prediction/` | Independently usable append-only ridge; versioned ordered feature/target/preprocessing/horizon contracts; fixed positive lambda; zero center; no forgetting or implicit intercept; all declared coefficients penalized |
| Ridge arithmetic/state | `A=lambda*I+sum(xxT)`, `b=sum(xy)`, checked small Cholesky solve; no inverse or jitter; complete typed statistics/coefficients/counts/binding; explicit numerical/capacity refusal; failed append keeps original bytes |
| `coleman/optimization/continuous*.py`, `_continuous_*.py` | Separate small SPD quadratic capability, Euclidean projection and projected online steps; one through four coefficients; bounded rational active-face work and checked binary64 KKT emission |
| Feasible set | Finite coordinate boxes; simplex or signed-box profile; explicit net and implied cash `1-sum(x)` intervals; optional gross and leverage ceilings; explicit dimensionless coefficient semantics |
| Result semantics | Proposed versus accepted vector, original problem/comparator/transition binding, named tolerances, primal/stationarity/complementarity diagnostics; typed optimal/infeasible/limited/unsupported/numerical_failure |
| `examples/continuous/` | Real chronological feature/predict/eligible-label flow through existing retained causal replay; separate constrained solver and finite solver; fixed/mean/last forecast controls and fixed/periodic/finite/continuous/projected allocation routes |
| Recovery tests/driver | Actual SQLite interruption before commit, identical/divergent concurrent writers, complete pending-label recovery, and one checkpoint with 32 real predictors plus 32 real Hedge learners |
| Human documentation | `docs/continuous-decisions.md`, API page, input/output table, worked arithmetic, primary literature links, Mermaid lifecycle and connected README/navigation; generated `llms.txt` updated |

Existing finite allocation, the original `optimization/mixture.py`, native CI
behavior and old SDK resources were preserved. Do not replace those solvers or
silently change old meanings. No dependencies or package version were changed.

Important callable entry points:

```python
from coleman.prediction import AppendOnlyRidge, RidgeConfig, RidgeInput, RidgeSample
from coleman.optimization.continuous import (
    solve_continuous, project_continuous, projected_update,
)
```

Use source and examples for complete constructor fields, including explicit
`schema_version=1` in nested continuous records. Minimal math imports no CLI,
MCP, native runner, storage, replay, NumPy or SciPy. Optional examples own their
source capture, pending labels, delivery and storage outside those kernels.

## Scientific and compatibility limits to preserve

- Signed outcomes, zero, null, unavailable and not-applicable are distinct.
  Zero labels change the Gram matrix; missing labels are never imputed.
- Direct mathematical append is intentionally non-idempotent. Existing replay
  owns accepted-event deduplication and `preserve_live_v1`: later corrections
  audit rather than retracting an already applied training label.
- Retain original pre-decision features and predictions while labels are
  pending. New preprocessing, feature schema or resets need explicit new lineage.
- The accepted ridge prefix, not a later full sample or corrected history, is
  the independent batch-fit comparison. Linear scores are not calibrated
  probabilities; sufficient statistics are not calibrated uncertainty.
- The qualified optimizer is small and **strictly positive definite**. Singular
  PSD, indefinite, nonconvex and arbitrary nonlinear formulations are unsupported.
  Exhausted face work is limited, not proof of infeasibility. No clipping,
  renormalization, hidden regularization or global nonconvex exactness claim.
- Signed mathematical coefficients do not qualify shorting, funding, collateral,
  margin, physical trades or consumer risk. Keep domain mechanics outside math.
- Projection faces/solve iterations are not online observations. External regret
  on the same realized functions does not bound comparator-specific net wealth,
  turnover or risk. Objective penalties are not paid charges.
- The four-step ridge example loses to the fixed zero prediction: MSE about
  3.08674 versus 1.5. Preserve the result and insufficient-evidence status.
- The active coupled fixture returns `(0.5,1.25)`, score `-2.8125`;
  coordinate clipping gives feasible `(0.5,1)`, score `-2.75`.
- Unknown versions reject. New artifacts do not imply silent migration or
  executable continuation of v4 checkpoints. Preserve the existing v4 readers.

## Validation actually performed

The portable [evidence summary](handoff/issue-365-evidence.json) records the
pre-handoff 47-file snapshot and historical author-seal digests. These are
scoped evidence, not final integration approval.

Fresh checks while preparing this handoff, against the current implementation:

- **170 passed**, zero skips/failures, with `--no-cov`: all 69 prediction cases,
  89 continuous optimizer cases and 12 source flow/regression cases.
- Whole-repository Ruff lint and format check passed (968 Python files formatted).
- NumPy docstring parameter audit passed: 15 files, 44 signatures.
- Actual generated `llms.txt` equality and `git diff --check` passed.

Historical scoped checks before the pause:

- Prediction: 69 passed and ten scoped quality checks passed; all 13 sealed
  source/test/golden files still matched the seal at handoff.
- Continuous optimization: 89 passed and ten scoped quality checks passed; all
  15 sealed files still matched the seal at handoff.
- Recovery: 8 passed in 97.18 seconds, including actual process death,
  concurrency and 64-component state. Its two files still match the seal, but
  **model.py and flow.py changed afterward**; rerun on the final context.
- Noneditable prototype: actual prediction/solve/pending resume, cold base
  imports, 32 predictors + 32 Hedge recovery and interrupted SQLite commit
  passed. This precedes the last model/provenance fixes; **it is not final
  installed-package qualification**.
- Prior static preflight passed Ruff, format, Pyright, Ty, full Pylint on the new
  SDK/examples and NumPy. Strict docs also passed on an earlier snapshot.

Two confirmed review findings were corrected before transfer: synthetic label
provenance now binds its actual `flow.py` generator, and a completed prediction
cannot claim a prefix containing its own or a later accepted label. Fresh
regressions pass in the 170-case run. Independent re-review is still needed.
The earlier feature-substitution allegation was retracted after a real probe;
the current original-feature joins reject it. Do not carry it as an open defect.

**Not done:** complete independent review, final complete merge-base/local
delegation review, current full-suite execution/raw coverage gate, final source
freeze/archive parity/noneditable qualification, PR, CI/review resolution, merge,
issue closure or release. No raw 95% coverage claim is made for this WIP.

## Finish #365 when assigned

1. Import this commit/branch; verify its source manifest and refresh public
   issue/dependency context. Do not start another implementation in parallel.
2. Finish independent source/math/temporal/state review. Reproduce findings,
   fix confirmed critical/high/medium problems with meaningful regressions,
   and re-review material changes. Confirm every acceptance criterion against
   actual code. Review excluded tests, Markdown, golden data and generated docs.
3. Run final lint, format, Pyright, Ty, Pylint, NumPy docstrings, Interrogate,
   Radon/Xenon at existing thresholds, strict docs, generated-doc equality,
   locked dependency checks, build/Twine and archive/source parity. Pylint4.1.3
   and Twine7.0.0 were available locally beyond the locked dev extras.
   Do not turn historical diagnostics in untouched modules into a claimed pass;
   compare actual baseline diagnostics and fix new confirmed problems.
4. Freeze exact source/index bytes; build a noneditable wheel. Exercise base
   cold imports, prediction-only, separate real constrained solve, original
   pending-label continuation, real interrupted/concurrent store operations,
   32 predictor + 32 Hedge state and CI/TCPCI compatibility outside the checkout.
   Optional `COLEMAN_TEST_CONTINUOUS_WHEEL` and existing
   `COLEMAN_TEST_LEARNING_WHEEL` accept a supplied matching wheel.
5. Collect the entire test inventory, preserving all prior **8,833** base test
   IDs plus the new cases. Run the complete suite with fresh coverage, original
   four workers/`loadfile` and the existing two serial corpus modules. Keep
   original test, boot, resource, timeout and CI fences. Enforce raw combined
   coverage **>=95%** without rounding, denominator changes, extra omissions,
   removed tests or coverage unions from failed/older source runs.
6. Run **Alibaba OpenCodeReview 1.12.13, local delegation only** on the entire
   main and `release/v5` merge-base ranges, plus workspace preview while dirty.
   Retrieve applicable rules for every returned path. The initial workspace
   preview covered 47 files (22 reviewable, 25 excluded) and three rule groups;
   it is not final review authority. Include the new handoff files too.
7. Only after complete local qualification, make cohesive commits, consolidate
   fixes and push. Open one PR targeting **`release/v5`**, linked to #365, with
   contract/architecture/compatibility/examples/migration/test evidence,
   limits/deferred decisions and actual local-review receipts. Do not declare
   the WIP evidence final.
8. Wait for reviews and required checks; reproduce valid feedback locally,
   add regressions, review the full diff again and push consolidated fixes.
   Respond with commit/evidence and resolve threads. Merge normally under the
   protected-branch policy, without bypassing checks or leaving confirmed
   critical/high/medium findings (including P2/medium) unresolved.
9. Then close #365 with acceptance evidence, update public Project 3/parent,
   sync local integration state and remove only the integrated feature branch.

Useful commands (after provisioning the locked environment and required tools):

```bash
uv sync --frozen --extra ci --extra analytics --extra dev --extra docs --extra mcp --extra release --group build --no-build --no-install-project
uv pip install --python .venv/bin/python --no-deps --no-build --no-build-isolation --editable .
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pyright --pythonpath .venv/bin/python coleman
.venv/bin/ty check coleman tests --output-format concise
make docstrings
.venv/bin/pylint --max-line-length=120 coleman/prediction coleman/optimization examples/continuous
.venv/bin/pylint --errors-only coleman
.venv/bin/interrogate coleman -v
.venv/bin/xenon --max-absolute C --max-modules B --max-average A coleman
make cost-maintainability
.venv/bin/zensical build --strict
uv lock --check --offline

# Focused, current integration checks; deliberately no coverage claim:
.venv/bin/python -m pytest tests/prediction tests/optimization tests/continuous_integration --no-cov

# Complete fresh parallel phase, then append only this run's serial completion:
.venv/bin/python -m pytest -n 4 --dist=loadfile --ignore=tests/corpus/test_installed_public_ci.py --ignore=tests/corpus/test_installed_westermo.py --cov=coleman --cov-branch --cov-fail-under=0
.venv/bin/python -m pytest tests/corpus/test_installed_public_ci.py tests/corpus/test_installed_westermo.py --cov=coleman --cov-branch --cov-append --cov-fail-under=95
.venv/bin/coverage json -o coverage-final.json
.venv/bin/python -c 'import json; t=json.load(open("coverage-final.json"))["totals"]; assert 100*(t["covered_lines"]+t["covered_branches"]) >= 95*(t["num_statements"]+t["num_branches"])'

# Use cached/offline npm execution when available. No external model endpoint.
npx --yes @alibaba-group/open-code-review@1.12.13 delegate preview --format json --from main --to HEAD -b 'Implement issue #365: independent online ridge and continuous allocation with explicit units, causal recovery, independent oracles and deterministic evidence'
npx --yes @alibaba-group/open-code-review@1.12.13 delegate preview --format json --from release/v5 --to HEAD -b 'Implement issue #365: independent online ridge and continuous allocation'
# Dirty workspace: delegate preview --format json -b '...'; no --workspace flag.
# For every actual preview path: delegate rule --format json -b '...' PATH...
git diff --check
```

The old public HCS source cache and local ignored receipts are not required
imports or source assets and are not in Git. Recreate any needed cache using the
existing pinned public-source workflow. Never convert missing source evidence
into zero, a skip that changes required coverage, or admission/efficacy claims.

## Transfer and portable evidence

**A ZIP is optional.** With this repository connected to its environment, Cloud
can use the branch/commits available on GitHub; see the
[official cloud environment guide](https://learn.chatgpt.com/docs/environments/cloud-environments).
Select/fetch the feature branch, rather than assuming the default/main checkout
contains the WIP. A local commit does not become available on GitHub by itself.

At this follow-up, `e5d7905` and the working-agreement update are local only;
`git ls-remote` returned no published feature-branch head. No push is part of the
handoff. The user requires full review/qualification before any push; a WIP
transfer commit does not waive that rule. Do not silently publish it to avoid
using a bundle. Once the branch is legitimately published, GitHub is the normal
transfer route and the code bundle is unnecessary.

If the feature branch is available on GitHub:

```bash
git fetch origin main release/v5 feat/v5-365-continuous-prediction
# Only if this local branch name does not already exist:
git switch --track -c feat/v5-365-continuous-prediction origin/feat/v5-365-continuous-prediction
```

For a local-only transfer, the optional ZIP contains a thin Git bundle and
whitelisted public-only evidence. After extracting it, fetch the prerequisite
integration branch and import the bundle, using an isolated checkout if the
feature branch already exists. Verify the bundle's head: an older ZIP is a
snapshot and will not acquire later documentation commits automatically.

```bash
git fetch origin release/v5
git bundle verify /path/to/branch.bundle
git fetch /path/to/branch.bundle refs/heads/feat/v5-365-continuous-prediction:refs/remotes/handoff/issue-365
# Only if this local branch name does not already exist:
git switch -c feat/v5-365-continuous-prediction refs/remotes/handoff/issue-365
```

The local ignored baseline main-review ledger is provided in the transfer ZIP,
not as fresh #365 authority. It covers 683 prior paths from the completed #369
baseline. Reuse only exact source bytes **and** identical applicable-rule hashes,
with explicit historical authority; re-review changed overlapping files and
new files. If an authority cannot be verified, review the relevant range again.
Do not claim a fresh historical reread or complete review from a summary hash.

GitHub also does not contain the ignored local `.git/coleman-v5-planning`
receipts, old runtime caches or personal local skills. The committed evidence
manifest identifies their scope; the optional archive carries selected receipts.
They are historical evidence, not required credentials or a substitute for new
verification. Without them, independently reproduce/review the affected scope
rather than inventing prior authority. A package is not a requirement of the
development workflow.

## Suggested Cloud assignment

Use this only when actually delegating continuation of the paused implementation:

> Work on public `jacksonpradolima/coleman`, branch
> `feat/v5-365-continuous-prediction`, targeting `release/v5`. Read the root
> `AGENTS.md`, `HANDOFF-365.md`, the evidence manifest, public issue #365 and all
> comments, parent #353 and relevant public Project 3 dependencies. Verify that
> the checkout contains the transferred WIP before changing anything. Finish
> #365 according to the working agreement: independent mathematical/temporal/
> state review; real recovery, concurrency and noneditable installed flows;
> complete unchanged quality and raw coverage gates; Alibaba OpenCodeReview
> **1.12.13 local `delegate preview`/`delegate rule` only**, both full merge-base
> ranges plus dirty workspace, manual excluded-file review and 100% changed-file
> coverage. Fix confirmed critical/high/medium findings with regressions and
> consolidated commits/pushes. Then open and follow the PR through required
> reviews/checks and normal protected merge into `release/v5`, closing #365 only
> with acceptance evidence. Preserve generic/public scope and all scientific,
> compatibility and release limits. Continue autonomously within this assigned
> issue; stop after #365 and report the next ready dependency. Do not publish
> v5 or begin another implementation.

## Roadmap after this issue

The remaining required order at the audit was:
**#365 → #370 → #366 → #378 → #379 → #380 → #372 → #381**.
Some are independently ready; this serial order respects the parent sequencing
and actual dependencies. Refresh the public graph before picking a successor.
#374 is optional v5.x research with an entry gate still pending; its completed
technical prerequisites do not authorize an advanced learner implementation.
This handoff starts none of those tasks.
