# COLEMAN agent working agreement

This file records the repository workflow for local and cloud agents. Follow
the user's current assignment and any applicable nested instructions. A user
request to pause or prepare a handoff does not resume implementation. Work
autonomously within the authorized task; routine edits, checks and review fixes
do not require repeated permission requests. Do not invent approval gates.

For the unfinished #365 snapshot, read [HANDOFF-365.md](HANDOFF-365.md) and
[its evidence manifest](handoff/issue-365-evidence.json) before touching code.
Historical passes or a WIP commit do not certify current integration readiness.

## 1. Establish scope and the correct base

- Inspect branch, remotes, working tree, main, integration branch and current
  release. Preserve user changes; never reset, force-push or discard work to
  make a check pass. Use an isolated worktree when needed.
- Read the entire assigned public issue, all comments, parent, public roadmap
  dependencies, applicable instructions and actual implementation/tests/docs.
  Confirm that the issue still fits the code. Reuse existing capabilities.
- A planned or Designed item is not implemented. Illustrative issue APIs are
  not authoritative; confirm the real interfaces and prerequisite completion.
- Implement one dependency-ready issue at a time. Create focused branches from
  updated `release/v5`; v5 implementation PRs target **`release/v5`**, not main.
  Independent agents may handle isolated files or reviews within that issue.
- Keep one coordinating owner for Git/index operations, global builds and
  coverage runs. Assign explicit file ownership; do not edit another active
  author's files. Distinguish author checks from independent peer review.
- Do not create replacement issues when an existing issue can contain the fix.
  Create an issue only for a real otherwise-unresolvable blocker.

## 2. Public scope, architecture and scientific behavior

- This repository is public. Consult only public COLEMAN sources and the public
  v5 [Project 3](https://github.com/users/jacksonpradolima/projects/3). Do not
  search, infer or copy nonpublic consumer/project details, names, examples,
  schemas, datasets, prompts or decisions. Use generic or CI/TCPCI examples.
- Preserve TCPCI and existing artifact meanings. Keep new mathematical cores
  independent of CLI, MCP, runners, providers and storage. Reuse the existing
  replay, state, solvers and task application instead of rebuilding them.
- Use explicit typed, versioned contracts, units and semantics. Preserve zero,
  permitted negatives, null, unknown, unavailable and not-applicable separately.
  Reject nonfinite and unsupported future versions. Separate logical identity,
  deterministic content fingerprints, provenance and source authentication.
- Define compatibility, old readers and explicit migration/continuation limits.
  Never silently reinterpret old results. Distinguish reading an artifact from
  executing/resuming a checkpoint. Restore every owned statistic, counter, RNG
  buffer, cache, pending label and domain path required by the selected route.
- Preserve proposed versus executed actions and original-producer feedback.
  Distinguish direct non-idempotent math from session delivery deduplication,
  durable acknowledgement and actual external effects. No unqualified
  exactly-once physical execution claim.
- Use primary papers/official documentation for mathematical assumptions.
  Supply independent worked arithmetic/oracles and simple matched baselines.
  Negative and inconclusive results are valid; no adaptive win is required.
  Seeds are not independent observations. Ranking, reward, mixture coefficients,
  calibrated probability and inference are different quantities.
- Do not transfer reference regret guarantees to controlled/delayed/costly
  variants without their assumptions. Do not call a method exact outside its
  qualified mathematical and numerical envelope. Prefer explicit unsupported,
  infeasible, limited, insufficient-evidence or conservative fallback outcomes.
- Separate model/learner loss, solver objective, coefficient movement, executed
  holdings/turnover, physical work and paid costs. Mechanical coefficient
  constraints do not qualify consumer risk, holdings, funding or trades.

## 3. Implement, document and test the real boundary

- Keep changes focused. Prefer small composable components over broad refactors.
  Public docstrings use NumPy style with meaningful Parameters, Returns, Raises
  and Notes as applicable: units, invariants, side effects, compatibility and
  limits. Write connected human docs with complete input/cycle/output examples,
  primary citations and honest capability/status distinctions. Update Mermaid
  diagrams when architecture changes.
- For new contracts/state, test deterministic round trips, versioned independent
  golden fixtures, existing readers, unknown versions, unit incompatibility,
  zero/null/unavailable, permitted negatives, nonfinite values, identity/digest,
  meaningful property/metamorphic behavior and numerical failures as applicable.
- Stateful work needs actual duplicate/idempotent delivery, interruption,
  checkpoint continuation and concurrent writer tests. Domain flows need at
  least one real end-to-end execution without mocking the principal component.
  Test import boundaries and noneditable installed packages outside the checkout.
- Keep independent mathematical oracles independent of production helpers.
  Never regenerate expected values from the implementation merely to pass.
- Update generated `llms.txt` through `scripts/generate_llms_txt.py`, then check
  equality. A generated file needs source/diff and generator verification,
  not a claim of an invented full manual reread.

## 4. Validate a frozen candidate

Provision from `uv.lock`/`pyproject.toml` and current CI; the current supported
Python is 3.14 or newer, with CI on 3.14. Use the chosen `.venv/bin/python`
directly; resolving its symlink can accidentally select a global environment.
Install required optional extras/tooling for the task. Node/npm/npx and the
exact Alibaba package below are also review prerequisites. A local personal
skill, npm cache or ignored receipt directory is not automatically in Cloud.

Before a qualified push, run the relevant complete repository gates: Ruff lint
and format, Pyright, Ty, actual Pylint, NumPy parameter audit, Interrogate100,
Radon maintainability>=20 and Xenon C/B/A, strict documentation, generated-doc
equality, frozen lock, package build/Twine and wheel/sdist/source parity, installed
smoke/end-to-end tests and `git diff --check`. Current configuration and task
requirements determine exact commands; the handoff has the #365 command list.
Do not assume that a workflow's filename or pre-commit covers all these checks.

Freeze and record the exact candidate sources/tree and artifact hashes. Tests,
package checks, manual review and required CI must describe the same candidate.
After material changes, rerun affected gates and re-review; identify any exact
unchanged evidence being reused. A receipt without matching source/context is
historical evidence, not a fresh pass. Preserve failures rather than combining
failed/old coverage into a passing candidate.

For complete coverage qualification, preserve all existing collected tests and
the current coverage denominator, including subprocess/thread/multiprocessing
measurement and executed SDK resource extensions. Use the established four
workers with `--dist=loadfile`. Only these original two corpus modules complete
serially after the fresh parallel phase:

- `tests/corpus/test_installed_public_ci.py`
- `tests/corpus/test_installed_westermo.py`

Append only that same candidate/run's serial coverage. Enforce the actual
unrounded combined **>=95%** threshold. Keep original test, boot, memory, work,
timeout and CI fences. Do not delete tests, add omissions, loosen limits or
extend deadlines to force green. Resolve missing tooling/source/network access
honestly; do not silently convert a required check to a skip. Local/file review
coverage and code line/branch coverage are separate quantities.

## 5. Mandatory local OpenCodeReview before every qualified push

Use **`@alibaba-group/open-code-review@1.12.13`**, **local delegation only**:
`delegate preview` and `delegate rule`. Never configure a model/API endpoint,
send source to an external review service, or use another package mode as a
substitute. Fetching the pinned npm tool is setup, not permission to export code.
Once cached, use `npx --offline` to make local execution explicit.

**Preview/rule output is not an automatic code review or approval.** Preview
describes changed files/exclusions; rule returns the criteria. The agents must
actually read the complete diff, apply every relevant rule and inspect code.

Review the entire actual merge-base ranges from both main and `release/v5` to
the candidate HEAD. Fetch their remote heads and verify the local base refs are
current; fetching alone does not fast-forward an existing local branch. Never
reset another worktree's branch to align it. If local refs cannot safely match,
use explicit freshly fetched remote refs and record the actual bases instead.
The commands below assume the named local base refs have been verified.
If the workspace is dirty, also run workspace preview;
ensure new/untracked implementation files are included (inspect/stage the
authorized files first). Do not use a nonexistent `--workspace` flag.

```bash
# Setup/cache the exact tool before offline review; help is not review evidence.
npx --yes @alibaba-group/open-code-review@1.12.13 delegate preview --help

export COLEMAN_REVIEW_CONTEXT='Implement the assigned public COLEMAN issue: actual problem, contract, compatibility and qualification'
export COLEMAN_REVIEW_DIR="$(mktemp -d)"

npx --offline --yes @alibaba-group/open-code-review@1.12.13 delegate preview --format json --from main --to HEAD -b "$COLEMAN_REVIEW_CONTEXT" > "$COLEMAN_REVIEW_DIR/main-preview.json"
npx --offline --yes @alibaba-group/open-code-review@1.12.13 delegate preview --format json --from release/v5 --to HEAD -b "$COLEMAN_REVIEW_CONTEXT" > "$COLEMAN_REVIEW_DIR/integration-preview.json"

# Run this as well whenever there are workspace changes:
npx --offline --yes @alibaba-group/open-code-review@1.12.13 delegate preview --format json -b "$COLEMAN_REVIEW_CONTEXT" > "$COLEMAN_REVIEW_DIR/workspace-preview.json"

# Retrieve rules for the union of actual reviewable AND excluded paths.
python3 - <<'PY'
import json
import os
from pathlib import Path
import subprocess

directory = Path(os.environ['COLEMAN_REVIEW_DIR'])
paths = set()
for source in directory.glob('*-preview.json'):
    data = json.loads(source.read_text())
    paths.update(row['path'] for key in ('reviewable_files', 'excluded_files') for row in data[key])
if not paths:
    raise SystemExit('No changed paths found; verify refs and workspace inclusion')
command = [
    'npx', '--offline', '--yes', '@alibaba-group/open-code-review@1.12.13',
    'delegate', 'rule', '--format', 'json', '-b', os.environ['COLEMAN_REVIEW_CONTEXT'],
    *sorted(paths),
]
with (directory / 'rules.json').open('w') as output:
    subprocess.run(command, stdout=output, check=True)
PY
```

Use the real issue description in the context. Preserve actual commands, exit
codes, complete diff, Git status/numstats, preview, rule groups and source hashes.
Manually review excluded files, especially tests, Markdown and golden JSON
semantics. The default JSON rule alone does not validate scientific fixtures.
Record path-by-path coverage: source hash, applied rule hash, reviewer/authority,
findings and disposition. Require **100% changed-file review**, distinct from
line coverage. Reuse older review only for exactly identical bytes and applicable
rules with verifiable explicit authority; otherwise review again. Independent
review does not mean an author restating their own test/seal.

Reproduce and fix every confirmed critical/high/medium problem, with appropriate
regressions. Re-run the review after material fixes. Do not report speculative
style suggestions as confirmed correctness defects. A missing review tool or
incomplete review is a reported pending gate, never a successful substitute.

## 6. GitHub delivery, review and completion

- Make cohesive conventional commits. Consolidate locally confirmed fixes;
  avoid a micro-push for each comment. Push only after the complete local gate
  and review above. A requested local WIP/handoff commit is permitted, but is not
  an approved integration/publication and does not waive the pre-push contract.
- For an assigned implementation authorized through delivery, open a PR linked
  to the existing issue, targeting `release/v5`. Describe the final problem and
  behavior, contract, architecture/compatibility, complete usage, migration,
  actual tests/commands, limits/deferred decisions and local review evidence.
- Wait for actual required checks and reviews on the exact current head. For
  valid comments: reproduce/check locally, fix with regression, repeat full-diff
  review and gates appropriate to the change, push consolidated fixes, reply
  with commit/evidence and resolve the thread. Do not claim an unavailable,
  superseded or stale review/check approved the current candidate.
- Merge only when repository policy, required checks/reviews and unresolved
  blocking comments permit it. No admin/bypass/forced merge. Use the repository's
  permitted normal merge method and exact-head protection where available.
- Then update the issue acceptance evidence and public roadmap/parent, sync
  local integration state and remove only the merged feature branch. If scope
  remains, do not mark the issue complete. Follow the next dependency only when
  the current user assignment authorizes continuing.
- Keep v5 issue integration separate from publication: no automatic v5 bump,
  intermediate stable/prerelease, tag, package or main merge. The single final
  v5 qualification/release belongs to #381 and its release policy.

## Code Review Rules

Prioritize reproducible correctness/security/scientific defects. Flag future
information entering training/tuning, substituted producer features or labels,
false probability/uncertainty/independence claims, incomplete continuation,
unacknowledged state advertised as committed, ambiguous units/missingness,
objective penalties reported as fees, silent old-data reinterpretation and
numerical/infeasible outcomes falsely reported as accepted/optimal. Confirm the
actual typed boundary and call context before raising a finding. Preserve
correct negative/inconclusive results and honest unsupported outcomes.
