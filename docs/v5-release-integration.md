# Integrate v5 once, publish when qualified

COLEMAN v5 is developed on **`release/v5`**. Each issue has a focused branch and a
reviewed PR targeting that integration branch. CI verifies each PR and the
integrated result. It does not create intermediate GitHub releases or publish the
development documentation over the stable site. A final reviewed PR integrates the
qualified work into `main`, where the existing release workflow publishes **5.0.0**.

This is a deliberate release-branch strategy for an incompatible change, rather
than a requirement for every project. It trades a controlled publication boundary
for the work of synchronizing fixes and reviewing the complete integration diff.
The [architecture specification](v5-architecture.md) and
[compatibility inventory](v5-compatibility.md) define what is intended to ship;
their proposed contracts become supported only after their implementation gates.

## Work from the integration branch

Start with a clean checkout. For example, the contracts issue would use:

```bash
git fetch origin --prune
git switch release/v5
git pull --ff-only origin release/v5
git switch -c feat/v5-355-decision-contracts
# Implement, test, review, then commit and push the issue branch.
gh pr create --base release/v5 --head feat/v5-355-decision-contracts --body-file /tmp/coleman-pr.md
```

Open each implementation PR only after the complete local review described in
[Contributing](contributing.md). Use Alibaba OpenCodeReview **1.12.13**, exclusively
in local delegation mode. Replace the issue branch in this example:

```bash
npx --yes @alibaba-group/open-code-review@1.12.13 delegate preview \
  --format json --from release/v5 --to HEAD \
  -b 'Implement the current v5 issue against its integration base'
# Also review the complete accumulated v5 diff before every push:
npx --yes @alibaba-group/open-code-review@1.12.13 delegate preview \
  --format json --from main --to HEAD \
  -b 'Review the complete v5 integration against stable main'
# If there are uncommitted changes, also inspect the workspace:
npx --yes @alibaba-group/open-code-review@1.12.13 delegate preview --format json
```

Read the applicable `delegate rule` results for every returned file. Manually
review excluded files, including tests and Markdown. Record complete changed-file
review coverage separately from code line coverage. Keep the whole diff from
`main` before every push as the integration grows, in addition to the focused
issue diff; before final publication, review `main...release/v5` and run the full
qualification again. Do not configure an
external review endpoint.

Use explicit issue and PR links. GitHub closing keywords only take effect for a
PR targeting the default branch; `Closes #355` in a PR to `release/v5` does not
close it. When its acceptance criteria are met and its PR is merged, record
**integrated into `release/v5`; unreleased** with the merge commit and verification
evidence. Close the issue deliberately only when its own completion contract is
satisfied; closing an integration task does not claim a published capability.
The release epic remains open until final publication is verified.

## Keep stable maintenance independent

`main` continues to accept stable v4 fixes until the final v5 merge. A normal
compatible `fix:` can publish a v4 patch independently of unreleased v5 commits.
Bring applicable security and correctness fixes into `release/v5` through a
reviewed synchronization PR. Preserve history on the shared branch; do not rebase
or force-push it. Recheck compatibility fixtures and the final version calculation
after each synchronization. Never send partially qualified v5 changes back to
`main` to resolve drift.

The integration branch should block deletion and force-push, require PRs and
resolved review threads, and require the always-executed quality checks after
their exact names are verified on GitHub. Path-filtered security checks cannot be
required unconditionally without an always-running aggregate. They still must
pass whenever dependency changes trigger them. Repository settings are verified
separately from source configuration; this paragraph is the policy, not proof
that a particular remote rule is installed.

## Publication and CI boundaries

| Event | Tests and review | Release publication | Stable documentation deployment |
| --- | --- | --- | --- |
| Issue PR targeting `release/v5` | Tests of the PR merge, lint/docstrings, complexity, strict docs, titles, dependency review and generated docs; advisory audit on relevant paths | None | None |
| Push to `release/v5` | Tests and strict docs; advisory audit for dependency changes | None | None |
| Manual dispatch on `release/v5` | Read-only workflows may run | Release job refuses the ref | Docs build may run; upload/deployment refuse the ref |
| Stable maintenance push to `main` | Existing stable checks | Semantic stable version when warranted | Stable site |
| Final reviewed v5 merge to `main` | Complete qualification required before merge | One final 5.0.0 release | Qualified v5 documentation |

The release job checks **`github.ref == 'refs/heads/main'`** for both push and
manual dispatch. A similarly named branch or a tag is not a release target.
Both Pages artifact upload and deployment enforce the same main-only boundary.
Because GitHub expression equality ignores case, a shell check also requires the
exact `refs/heads/main` before publication or Pages upload/deployment. Mixed-case
refs cannot pass that second check.
These guards describe the workflows delivered on this integration branch.
The existing v4 `main` retains its previous workflow files until a reviewed
backport or final promotion; this change alone does not harden old refs.
The test workflow covers pushes and PRs; those can produce two observations for a
local branch PR, with the PR event checking the synthetic merge result. Their
check names include the event (`push` or `pull_request`), so integration protection
can require the PR result without accepting a push result under the same name.
Read-only decision benchmarks can be manually run on the integration branch and
preserve their observations as workflow artifacts without creating a release.

## Rehearse versions without publishing

Python Semantic Release is pinned to **10.5.3** in the lock. Its `branches` table,
not the obsolete singular `branch` setting, defines two exact regular-expression
matches: `^main$` for stable versions and `^release/v5$` for alpha rehearsal.
The prerelease group does **not** enable a release workflow for that branch.
Package metadata and the lock remain at the stable version during this first
configuration delivery; no 5.0 tag is created.

Install the locked tools and run the real-tool regression suite:

```bash
uv sync --frozen --extra dev --extra release --group build
uv run --frozen --no-sync python -m pytest tests/test_release_routing.py tests/test_release_notes.py --no-cov
```

The tests create independent temporary Git repositories with synthetic commits,
the actual project configuration and a baseline `v4.20.0` tag. They do not share
refs with the checkout, pass no credentials, disable pushing and VCS publication,
skip builds, and check that project bytes, refs and worktree remain unchanged.
The principal release tool is real, not mocked.

| Fixture history / explicit option | Expected candidate |
| --- | --- |
| Stable baseline plus `fix:` on `main` | `4.20.1` |
| Breaking commit on `release/v5` | `5.0.0-alpha.1` |
| Explicit major prerelease rehearsal using `alpha`, `beta`, `rc` | `5.0.0-alpha.1`, `5.0.0-beta.1`, `5.0.0-rc.1` |
| Tagged `4.20.1` maintenance synchronized into breaking integration | `5.0.0-alpha.1` |
| Maintained stable history plus merged breaking integration on `main` | `5.0.0` |
| `main-preview`, `master`, `release/v50`, `release/v5-fix`, feature branch | Rejected release group |

For a rehearsal in a **throwaway repository**, the command shape is:

```bash
semantic-release --noop version --print --major --as-prerelease \
  --prerelease-token rc --no-push --no-vcs-release --skip-build
```

`--noop` is global and comes before `version`. A forced `--major` alone can select
a stable version even on a prerelease branch; always use `--as-prerelease` for
these explicit rehearsals. These simulations exercise version selection, not
asset signing, remote publication, authentication or a user-installable
prerelease. Public alpha/beta/RC releases would require an explicit change to the
agreed single-publication policy.

## The final release remains a separate gate

When all required v5.0 deliveries are qualified, the final release task must:

1. Synchronize applicable stable fixes, freeze the final tree and review the
   complete diff from the current `main`.
2. Run all quality, security, compatibility, installed-package and real CI and
   synthetic allocation gates, retaining their evidence and limitations.
3. Preserve a truthful breaking-change declaration in the final merge history.
   If squashing, the final PR title/commit must retain `feat!:` or the equivalent
   breaking-change footer; earlier feature commits do not survive a squash.
4. Run the pinned version preview against the **actual proposed main history**,
   require **exactly `5.0.0`**, and verify that tag does not already exist. A newer
   maintenance release or any unexpected candidate requires investigation, not
   an automatic version override.
5. Commit reviewed en-US `docs/releases/5.0.0.md`, migrations and the final
   delivered-feature inventory. Describe completed work, not the continuing
   application roadmap or rejected research.
6. Merge only with required checks and review findings resolved. Let the stable
   workflow update metadata and lock together, tag, build, sign and publish.
7. Verify the rendered notes, wheel/source versions, lock, SBOM and attestations;
   update the planned draft without leaving a duplicate public release. Then
   synchronize local `main` and remove integrated branches according to policy.

Do not force a major increment in the general workflow or bump only
`pyproject.toml` now. The future release gate owns the final publication rehearsal
and assets; this configuration delivery qualifies branch selection and the
publication boundary.

## Reference behavior

- [Python Semantic Release multibranch releases](https://python-semantic-release.readthedocs.io/en/latest/concepts/multibranch_releases.html)
  explains independent branch groups; the regression tests use the pinned 10.5.3
  implementation rather than assuming the latest documentation is identical.
- [The pinned configuration implementation](https://github.com/python-semantic-release/python-semantic-release/blob/v10.5.3/src/semantic_release/cli/config.py)
  and [version command](https://github.com/python-semantic-release/python-semantic-release/blob/v10.5.3/src/semantic_release/cli/commands/version.py)
  define exact matching and forced prerelease behavior.
- [GitHub workflow branch filters](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onpull_requestpull_request_targetbranchesbranches-ignore)
  distinguish PR target filters from push refs.
- [GitHub issue linking](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)
  documents the default-branch requirement for closing keywords.
