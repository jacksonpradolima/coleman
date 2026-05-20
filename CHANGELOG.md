# CHANGELOG


## v2.3.0 (2026-04-21)

### Bug Fixes

- Improve error handling in cost-profile-pyspy target for py-spy profiling
  ([`197ff7c`](https://github.com/jacksonpradolima/coleman/commit/197ff7cf3da07fc73eebfa3f22c430927254bef4))

### Chores

- Update uv.lock for version 2.2.0
  ([`2b4ee2b`](https://github.com/jacksonpradolima/coleman/commit/2b4ee2b68f98bcb790e573a5de2ade367fa3ca8c))

### Features

- Add Wily file reporting and improve py-spy error handling
  ([`b4fa83b`](https://github.com/jacksonpradolima/coleman/commit/b4fa83b2688f74bc3dab14b48e0a04810a89a189))

feat: add Wily file reporting and improve py-spy error handling

- Add Wily file reporting targets and update documentation for custom file usage
  ([`53e5850`](https://github.com/jacksonpradolima/coleman/commit/53e5850398a5d2b918ff848e1c96ef975c2ccc1e))


## v2.2.0 (2026-04-21)

### Bug Fixes

- Add permissions block to code-cost workflow
  ([`39e0681`](https://github.com/jacksonpradolima/coleman/commit/39e06812f1fe313f5491d0e85e538d9dbf4f5e7b))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/5633ad91-543d-4dac-927d-66e9abe79ef4

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Add pipefail to radon MI step to prevent masking exit status
  ([`c578209`](https://github.com/jacksonpradolima/coleman/commit/c578209304e51674c35276f0b33cdf882d2f32db))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/be1c7929-9610-4c66-acb8-7c32947dfc27

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address review feedback — real workload, wily/pyRAPL deps, CI gates, workflow notebook
  ([`6fb80b0`](https://github.com/jacksonpradolima/coleman/commit/6fb80b02f7a1369a6be01cd9cf70d43dca75fa17))

- scripts/measure_energy.py: use actual coleman experiment run instead of toy example -
  pyproject.toml: add wily and pyRAPL as dev dependencies (no longer optional) - docs/code-cost.md:
  mark Wily and pyRAPL as included; document dual CI gates - .github/workflows/code-cost.yml:
  enforce Radon MI threshold (fail if any module < A); Xenon remains primary gate; CC report runs
  always for visibility - docs/workflow.py: add code cost evaluation section to e2e workflow
  notebook - Makefile: add cost-wily target

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/4507e4d5-5347-46d9-9ccd-3e405b276ef8

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Apply review feedback — sys.executable, clean returns, script path, docs
  ([`92888ba`](https://github.com/jacksonpradolima/coleman/commit/92888baf4de4dad78d9e5ae971c19da255b51de8))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/e234b086-5883-4d02-993b-849c90699f90

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Apply review suggestions — pin Python 3.14, handle radon errors, try/finally tracker, lowercase
  pyrapl
  ([`bd23c00`](https://github.com/jacksonpradolima/coleman/commit/bd23c00802d1c494dcd47030d924c8ba695a6294))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/2ee4097d-694c-4fe1-a8bb-31d5df86aaca

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Correct spelling of 'analysing' to 'analyzing' in workflow.md
  ([`9d48ce0`](https://github.com/jacksonpradolima/coleman/commit/9d48ce02316a179b87ebdbeac01182d16808c3da))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/3186cde3-5a74-4e14-97a0-8f89d82d04c1

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Enforce MI threshold in cost-maintainability target to match CI
  ([`bbd0d83`](https://github.com/jacksonpradolima/coleman/commit/bbd0d83b9860da5eb3a372b58ccbab42d2158d3c))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/99984111-1110-4745-a017-1044a8721aa2

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Parse radon MI JSON defensively and fix workflow.md comment
  ([`ebdd696`](https://github.com/jacksonpradolima/coleman/commit/ebdd69685bb2307eb92e7db6a4e4b8faf100ad3b))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/774bf8da-1e43-45c4-b066-9442d24b9837

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Pin checkout hash, add docs-export-workflow target, update README and CONTRIBUTING with code cost
  ([`0259972`](https://github.com/jacksonpradolima/coleman/commit/025997281d40f762d0e797fba75d91b941fdd807))

- Use commit hash (de0fac2e) for actions/checkout in code-cost.yml - Add `make docs-export-workflow`
  to auto-generate workflow.md from marimo notebook - Add Code Cost Evaluation section to README.md
  (CI gates + make targets) - Add Code Cost Evaluation section to CONTRIBUTING.md (local checks + CI
  thresholds)

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/589c0583-815d-4ede-9ca4-873ec3e219c5

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Prevent partial mi_scores on unexpected radon format
  ([`5df95bf`](https://github.com/jacksonpradolima/coleman/commit/5df95bfa65b61edfe70379eeb6f9905776dfbe3f))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/774bf8da-1e43-45c4-b066-9442d24b9837

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Render workflow as markdown in Zensical instead of downloading .py
  ([`35736ee`](https://github.com/jacksonpradolima/coleman/commit/35736ee397e664ee1e0f7248d365d5eb64c2627a))

Zensical's Rust-based renderer doesn't execute Python plugins like mkdocs-jupyter, so .py/.ipynb nav
  entries were served as raw downloads.

- Add docs/workflow.md with all notebook content rendered as markdown - Update zensical.toml nav to
  point to workflow.md - Keep docs/workflow.py for interactive marimo use (marimo edit
  docs/workflow.py)

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/3186cde3-5a74-4e14-97a0-8f89d82d04c1

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Show actual radon error instead of always suggesting install
  ([`0f5df34`](https://github.com/jacksonpradolima/coleman/commit/0f5df3488fd2006021b88d4d8a31b43ddf818e6d))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/702ef268-24b6-4518-8a18-90dcde34ed83

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Use make commands in CI and fix scalene/pyspy/pyRAPL docs
  ([`b307fc4`](https://github.com/jacksonpradolima/coleman/commit/b307fc49cd214b73089318805d8ecac9d1d0ddaa))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/c04883f1-dfd1-49a5-a736-8b125de5c03f

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Chores

- Update uv.lock for version 2.1.0
  ([`d488102`](https://github.com/jacksonpradolima/coleman/commit/d4881025420122e2cb9c84da8f48cd356945bf6d))

- **deps**: Bump actions/github-script from 8.0.0 to 9.0.0 in /.github/workflows
  ([`66163b9`](https://github.com/jacksonpradolima/coleman/commit/66163b93bde8db91c8dc5980646d300e2f895221))

chore(deps): bump actions/github-script from 8.0.0 to 9.0.0 in /.github/workflows

- **deps**: Bump actions/github-script in /.github/workflows
  ([`437e77d`](https://github.com/jacksonpradolima/coleman/commit/437e77d5f6b1a47de2c827d2fe8a5f368893b721))

Bumps [actions/github-script](https://github.com/actions/github-script) from 8.0.0 to 9.0.0. -
  [Release notes](https://github.com/actions/github-script/releases) -
  [Commits](https://github.com/actions/github-script/compare/ed597411d8f924073f98dfc5c65a23a2325f34cd...3a2844b7e9c422d3c10d287c895573f7108da1b3)

--- updated-dependencies: - dependency-name: actions/github-script dependency-version: 9.0.0

dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump codelytv/pr-size-labeler from 1.10.3 to 1.10.4 in /.github/workflows
  ([`cf2339d`](https://github.com/jacksonpradolima/coleman/commit/cf2339dbe3938057f7e5eefb74301ec631775f17))

chore(deps): bump codelytv/pr-size-labeler from 1.10.3 to 1.10.4 in /.github/workflows

- **deps**: Bump codelytv/pr-size-labeler in /.github/workflows
  ([`877b425`](https://github.com/jacksonpradolima/coleman/commit/877b4250868b3ebb793b7284c00fe6a115c966a4))

Bumps [codelytv/pr-size-labeler](https://github.com/codelytv/pr-size-labeler) from 1.10.3 to 1.10.4.
  - [Release notes](https://github.com/codelytv/pr-size-labeler/releases) -
  [Commits](https://github.com/codelytv/pr-size-labeler/compare/4ec67706cd878fbc1c8db0a5dcd28b6bb412e85a...095a41fca88b8764fd9e008ad269bcdb82bb38b9)

--- updated-dependencies: - dependency-name: codelytv/pr-size-labeler dependency-version: 1.10.4

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump marimo from 0.22.5 to 0.23.0 in the uv group across 1 directory
  ([`53ec0f6`](https://github.com/jacksonpradolima/coleman/commit/53ec0f69177c03e54b0cc70d8ee42ba644012d09))

chore(deps): bump marimo from 0.22.5 to 0.23.0 in the uv group across 1 directory

- **deps**: Bump marimo in the uv group across 1 directory
  ([`36032db`](https://github.com/jacksonpradolima/coleman/commit/36032db263b3e872012fc0283924176d9ce1e4d9))

Bumps the uv group with 1 update in the / directory:
  [marimo](https://github.com/marimo-team/marimo).

Updates `marimo` from 0.22.5 to 0.23.0 - [Release
  notes](https://github.com/marimo-team/marimo/releases) -
  [Commits](https://github.com/marimo-team/marimo/compare/0.22.5...0.23.0)

--- updated-dependencies: - dependency-name: marimo dependency-version: 0.23.0

dependency-type: direct:production

dependency-group: uv

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump opentelemetry-api from 1.40.0 to 1.41.0
  ([`ac33be3`](https://github.com/jacksonpradolima/coleman/commit/ac33be30e228559dca53d88e35a2aba00fed05ed))

chore(deps): bump opentelemetry-api from 1.40.0 to 1.41.0

- **deps**: Bump opentelemetry-api from 1.40.0 to 1.41.0
  ([`713cc09`](https://github.com/jacksonpradolima/coleman/commit/713cc09a10d82d9f553e8217bba697850c4e02ad))

Bumps [opentelemetry-api](https://github.com/open-telemetry/opentelemetry-python) from 1.40.0 to
  1.41.0. - [Release notes](https://github.com/open-telemetry/opentelemetry-python/releases) -
  [Changelog](https://github.com/open-telemetry/opentelemetry-python/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/open-telemetry/opentelemetry-python/compare/v1.40.0...v1.41.0)

--- updated-dependencies: - dependency-name: opentelemetry-api dependency-version: 1.41.0

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump opentelemetry-exporter-otlp-proto-http
  ([`f37d837`](https://github.com/jacksonpradolima/coleman/commit/f37d837e812e9d285ea2ea2af16fd85eab2007d7))

Bumps
  [opentelemetry-exporter-otlp-proto-http](https://github.com/open-telemetry/opentelemetry-python)
  from 1.40.0 to 1.41.0. - [Release
  notes](https://github.com/open-telemetry/opentelemetry-python/releases) -
  [Changelog](https://github.com/open-telemetry/opentelemetry-python/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/open-telemetry/opentelemetry-python/compare/v1.40.0...v1.41.0)

--- updated-dependencies: - dependency-name: opentelemetry-exporter-otlp-proto-http
  dependency-version: 1.41.0

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump opentelemetry-exporter-otlp-proto-http from 1.40.0 to 1.41.0
  ([`dd83903`](https://github.com/jacksonpradolima/coleman/commit/dd839039d0bda95615dbd03bb87469d7975d0818))

chore(deps): bump opentelemetry-exporter-otlp-proto-http from 1.40.0 to 1.41.0

- **deps**: Bump pytest from 9.0.2 to 9.0.3
  ([`78ae841`](https://github.com/jacksonpradolima/coleman/commit/78ae8418ea0cef29c0845f09491983ca79a3afc6))

chore(deps): bump pytest from 9.0.2 to 9.0.3

- **deps**: Bump pytest from 9.0.2 to 9.0.3
  ([`7b4b47c`](https://github.com/jacksonpradolima/coleman/commit/7b4b47c8f49e7e157b85ab2d04a5e8f42a298a8c))

Bumps [pytest](https://github.com/pytest-dev/pytest) from 9.0.2 to 9.0.3. - [Release
  notes](https://github.com/pytest-dev/pytest/releases) -
  [Changelog](https://github.com/pytest-dev/pytest/blob/main/CHANGELOG.rst) -
  [Commits](https://github.com/pytest-dev/pytest/compare/9.0.2...9.0.3)

--- updated-dependencies: - dependency-name: pytest dependency-version: 9.0.3

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

### Code Style

- Simplify json import alias in module-level _parse_radon_mi
  ([`4001905`](https://github.com/jacksonpradolima/coleman/commit/400190587ce067be0f97f4dcf090fca4270b4b25))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/f3aacc22-849a-4932-b2cc-eee7d1b419ae

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Documentation

- Add example outputs and explanations to code-cost.md; fix scalene command; add cost-raw and
  cost-profile-pyspy make targets
  ([`584aab2`](https://github.com/jacksonpradolima/coleman/commit/584aab2d48fc9c8178de050d9a64cb1093ac6fd7))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/97882284-249d-4885-9cca-adb64e37ad30

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Features

- Add force_sequential_under_scalene option for Scalene profiling stability
  ([`aca6e74`](https://github.com/jacksonpradolima/coleman/commit/aca6e74e603ff09b0fad888a79f68afe0f73772c))

- Add Python 3.13 profiling workflow with py-spy in Makefile and update documentation
  ([`29ad762`](https://github.com/jacksonpradolima/coleman/commit/29ad7623f565581c09cb447dc44401f94f7fc3e6))

- Establish code cost evaluation workflow
  ([`5fbd76b`](https://github.com/jacksonpradolima/coleman/commit/5fbd76b35ed3f13c9007845e6956caddb675cf2f))

feat: establish code cost evaluation workflow

- Establish code cost evaluation workflow
  ([`0fa954e`](https://github.com/jacksonpradolima/coleman/commit/0fa954e279f73654483b3200a82cc5edab794aae))

Add radon, xenon, scalene, py-spy, and codecarbon as dev dependencies. Create
  scripts/measure_energy.py for energy estimation. Add GitHub Actions workflow for Xenon structural
  checks. Add Makefile targets for cost-structural, cost-complexity, cost-maintainability,
  cost-xenon, cost-profile-scalene, and cost-energy. Create docs/code-cost.md documenting all cost
  dimensions. Update .gitignore for CodeCarbon output files.

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/5633ad91-543d-4dac-927d-66e9abe79ef4

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Refactoring

- Move _parse_radon_mi to module level to reduce cell cognitive complexity to ≤15
  ([`062a2f8`](https://github.com/jacksonpradolima/coleman/commit/062a2f811256dc846c2160c599388b9896884653))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/f3aacc22-849a-4932-b2cc-eee7d1b419ae

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Reduce cognitive complexity in workflow.py structural cost cell
  ([`82126d8`](https://github.com/jacksonpradolima/coleman/commit/82126d8d1c29037b6833c35910ad4e3b422fb042))

- Extract MI parsing into local _parse_radon_mi helper function - Extract nested conditional
  expression into mi_passed variable and if/elif/else chain for mi_status - Extract xenon_status for
  consistency

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/312e716b-c96a-434b-b9d6-ab82002fe0c6

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Simplify getattr and remove overly generic error token
  ([`6ddf59c`](https://github.com/jacksonpradolima/coleman/commit/6ddf59c84f33283c9574e799afe30dba0d3de7d4))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/702ef268-24b6-4518-8a18-90dcde34ed83

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>


## v2.1.0 (2026-04-12)

### Bug Fixes

- Address code review feedback - use ValidationError, add durationUnit comment
  ([`87fa20f`](https://github.com/jacksonpradolima/coleman/commit/87fa20f49da330578a58ad9165ef4efad3b01e0e))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/341b9eed-a788-4671-8ae7-a61fd4d887b2

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address review feedback - Field(default_factory), service.name priority, pin plugin version,
  telemetry extra
  ([`1615e56`](https://github.com/jacksonpradolima/coleman/commit/1615e56ea328f030244c27ebb2fbd75e0d13b5d9))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/14da71b3-b350-4a22-aa27-32fab51710e0

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Always install Grafana ClickHouse plugin by default for devcontainer compatibility
  ([`7c8e683`](https://github.com/jacksonpradolima/coleman/commit/7c8e683d32d893c80b74592f00c4a6f6bd1e631e))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/8bbe7f0f-52dd-45f9-8c53-c3063dded884

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Gate GF_INSTALL_PLUGINS behind env var so base stack works offline
  ([`4106f19`](https://github.com/jacksonpradolima/coleman/commit/4106f196a1ef15e6e98a59ba9b33091dcd673940))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/c15c7cfa-0491-4096-8349-c1e21f13e5d8

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Chores

- Update uv.lock for version 2.0.0
  ([`ecc00b4`](https://github.com/jacksonpradolima/coleman/commit/ecc00b40dc4c76139217a869290b5d70a93ad4d4))

- **deps**: Bump actions/deploy-pages from 4.0.5 to 5.0.0 in /.github/workflows
  ([`1067d49`](https://github.com/jacksonpradolima/coleman/commit/1067d4985a444bc061c4a694fe57a17bbb97c68e))

chore(deps): bump actions/deploy-pages from 4.0.5 to 5.0.0 in /.github/workflows

- **deps**: Bump actions/deploy-pages in /.github/workflows
  ([`edb79c8`](https://github.com/jacksonpradolima/coleman/commit/edb79c831f60261c4d46fc2e1a5f4785beb2b678))

Bumps [actions/deploy-pages](https://github.com/actions/deploy-pages) from 4.0.5 to 5.0.0. -
  [Release notes](https://github.com/actions/deploy-pages/releases) -
  [Commits](https://github.com/actions/deploy-pages/compare/d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e...cd2ce8fcbc39b97be8ca5fce6e763baed58fa128)

--- updated-dependencies: - dependency-name: actions/deploy-pages dependency-version: 5.0.0

dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump anchore/sbom-action from 0.23.1 to 0.24.0 in /.github/workflows
  ([`dc2e26b`](https://github.com/jacksonpradolima/coleman/commit/dc2e26bf9bb81069b7640bd6cb618eeec3f495bd))

chore(deps): bump anchore/sbom-action from 0.23.1 to 0.24.0 in /.github/workflows

- **deps**: Bump anchore/sbom-action in /.github/workflows
  ([`67706c4`](https://github.com/jacksonpradolima/coleman/commit/67706c40a2861674e47853f6d9689b55b4215281))

Bumps [anchore/sbom-action](https://github.com/anchore/sbom-action) from 0.23.1 to 0.24.0. -
  [Release notes](https://github.com/anchore/sbom-action/releases) -
  [Changelog](https://github.com/anchore/sbom-action/blob/main/RELEASE.md) -
  [Commits](https://github.com/anchore/sbom-action/compare/57aae528053a48a3f6235f2d9461b05fbcb7366d...e22c389904149dbc22b58101806040fa8d37a610)

--- updated-dependencies: - dependency-name: anchore/sbom-action dependency-version: 0.24.0

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump codecov/codecov-action from 5.5.3 to 6.0.0 in /.github/workflows
  ([`993b3f4`](https://github.com/jacksonpradolima/coleman/commit/993b3f4e02daf1ca0547700095c9def23d54089e))

chore(deps): bump codecov/codecov-action from 5.5.3 to 6.0.0 in /.github/workflows

- **deps**: Bump codecov/codecov-action in /.github/workflows
  ([`ef6ebee`](https://github.com/jacksonpradolima/coleman/commit/ef6ebee342323925e0690e854f77150875ee488a))

Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 5.5.3 to 6.0.0. -
  [Release notes](https://github.com/codecov/codecov-action/releases) -
  [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/codecov/codecov-action/compare/1af58845a975a7985b0beb0cbe6fbbb71a41dbad...57e3a136b779b570ffcdbf80b3bdc90e7fab3de2)

--- updated-dependencies: - dependency-name: codecov/codecov-action dependency-version: 6.0.0

dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump marimo from 0.20.4 to 0.21.1
  ([`812bc40`](https://github.com/jacksonpradolima/coleman/commit/812bc403154b6b35f68d1aef61cddd9cce92254e))

chore(deps): bump marimo from 0.20.4 to 0.21.1

- **deps**: Bump marimo from 0.20.4 to 0.21.1
  ([`c190fde`](https://github.com/jacksonpradolima/coleman/commit/c190fde8881848106f9df5dc5a8c7716414fcfdc))

Bumps [marimo](https://github.com/marimo-team/marimo) from 0.20.4 to 0.21.1. - [Release
  notes](https://github.com/marimo-team/marimo/releases) -
  [Commits](https://github.com/marimo-team/marimo/compare/0.20.4...0.21.1)

--- updated-dependencies: - dependency-name: marimo dependency-version: 0.21.1

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump mkdocs-jupyter from 0.25.1 to 0.26.1
  ([`f4aff74`](https://github.com/jacksonpradolima/coleman/commit/f4aff745d3737551434996fec903446535ac9ee3))

chore(deps): bump mkdocs-jupyter from 0.25.1 to 0.26.1

- **deps**: Bump mkdocs-jupyter from 0.25.1 to 0.26.1
  ([`aa1751b`](https://github.com/jacksonpradolima/coleman/commit/aa1751beaf46e85a0411ec06888f87a536a095ba))

Bumps [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) from 0.25.1 to 0.26.1. -
  [Changelog](https://github.com/danielfrg/mkdocs-jupyter/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/danielfrg/mkdocs-jupyter/compare/0.25.1...0.26.1)

--- updated-dependencies: - dependency-name: mkdocs-jupyter dependency-version: 0.26.1

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pytest-cov from 7.0.0 to 7.1.0
  ([`e4e0b4a`](https://github.com/jacksonpradolima/coleman/commit/e4e0b4a6dfd2577f5780101d79062541203ff948))

chore(deps): bump pytest-cov from 7.0.0 to 7.1.0

- **deps**: Bump pytest-cov from 7.0.0 to 7.1.0
  ([`64dd1a9`](https://github.com/jacksonpradolima/coleman/commit/64dd1a911b4db0d3a35b5370c408cfc258f9e01b))

Bumps [pytest-cov](https://github.com/pytest-dev/pytest-cov) from 7.0.0 to 7.1.0. -
  [Changelog](https://github.com/pytest-dev/pytest-cov/blob/master/CHANGELOG.rst) -
  [Commits](https://github.com/pytest-dev/pytest-cov/compare/v7.0.0...v7.1.0)

--- updated-dependencies: - dependency-name: pytest-cov dependency-version: 7.1.0

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump requests from 2.32.5 to 2.33.0 in the uv group across 1 directory
  ([`8f6b5d8`](https://github.com/jacksonpradolima/coleman/commit/8f6b5d81808328b2142728abde883b0f5da44816))

chore(deps): bump requests from 2.32.5 to 2.33.0 in the uv group across 1 directory

- **deps**: Bump requests in the uv group across 1 directory
  ([`df1f5ab`](https://github.com/jacksonpradolima/coleman/commit/df1f5ab18ac4954b6f99ae2406239fd923eb9fba))

Bumps the uv group with 1 update in the / directory: [requests](https://github.com/psf/requests).

Updates `requests` from 2.32.5 to 2.33.0 - [Release notes](https://github.com/psf/requests/releases)
  - [Changelog](https://github.com/psf/requests/blob/main/HISTORY.md) -
  [Commits](https://github.com/psf/requests/compare/v2.32.5...v2.33.0)

--- updated-dependencies: - dependency-name: requests dependency-version: 2.33.0

dependency-type: indirect

dependency-group: uv

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump ruff from 0.15.7 to 0.15.8
  ([`400ae30`](https://github.com/jacksonpradolima/coleman/commit/400ae30071fd40f52af1e0eba8b525502c406c7f))

chore(deps): bump ruff from 0.15.7 to 0.15.8

- **deps**: Bump ruff from 0.15.7 to 0.15.8
  ([`169f867`](https://github.com/jacksonpradolima/coleman/commit/169f867e1e995485901adda053d614ecd3bd62a7))

Bumps [ruff](https://github.com/astral-sh/ruff) from 0.15.7 to 0.15.8. - [Release
  notes](https://github.com/astral-sh/ruff/releases) -
  [Changelog](https://github.com/astral-sh/ruff/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/astral-sh/ruff/compare/0.15.7...0.15.8)

--- updated-dependencies: - dependency-name: ruff dependency-version: 0.15.8

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump sigstore/gh-action-sigstore-python
  ([`f81a54b`](https://github.com/jacksonpradolima/coleman/commit/f81a54bdaa16760915a0a91b85d4bc61211102ce))

Bumps [sigstore/gh-action-sigstore-python](https://github.com/sigstore/gh-action-sigstore-python)
  from 3.2.0 to 3.3.0. - [Release
  notes](https://github.com/sigstore/gh-action-sigstore-python/releases) -
  [Changelog](https://github.com/sigstore/gh-action-sigstore-python/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/sigstore/gh-action-sigstore-python/compare/a5caf349bc536fbef3668a10ed7f5cd309a4b53d...04cffa1d795717b140764e8b640de88853c92acc)

--- updated-dependencies: - dependency-name: sigstore/gh-action-sigstore-python dependency-version:
  3.3.0

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump sigstore/gh-action-sigstore-python from 3.2.0 to 3.3.0 in /.github/workflows
  ([`a399392`](https://github.com/jacksonpradolima/coleman/commit/a3993924d1a129b8e0e5718b38c1664bfd7b2ba7))

chore(deps): bump sigstore/gh-action-sigstore-python from 3.2.0 to 3.3.0 in /.github/workflows

### Features

- Enable optional ClickHouse backend for OpenTelemetry + Grafana ClickHouse datasource
  ([`f2cd059`](https://github.com/jacksonpradolima/coleman/commit/f2cd05925b9fa1ac8bc23bfefa35607c3ee5c9b7))

feat: enable optional ClickHouse backend for OpenTelemetry + Grafana ClickHouse datasource

- Enable optional ClickHouse backend for OTel telemetry + Grafana datasource
  ([`8cf2a78`](https://github.com/jacksonpradolima/coleman/commit/8cf2a785e1e47d4320c47fdc126b253b0d94f6f5))

- Create otel-collector-config-clickhouse.yaml with ClickHouse exporter for metrics, traces, and
  logs pipelines - Update docker-compose.yaml to support config file selection via
  OTEL_COLLECTOR_CONFIG env var and install Grafana ClickHouse plugin - Provision ClickHouse
  datasource in Grafana with logs/traces config for Explore correlation workflows - Add
  resource_attributes parameter to Telemetry and get_telemetry() for passing execution_id/run_id as
  OTel resource attributes - Add resource_attributes field to TelemetrySpec model - Pass
  resource_attributes from Environment to get_telemetry() - Update observability README with
  ClickHouse mode documentation - Add tests for new resource_attributes functionality - Update
  golden run_id hash for new TelemetrySpec schema

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/341b9eed-a788-4671-8ae7-a61fd4d887b2

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>


## v2.0.0 (2026-04-08)

### Bug Fixes

- Accumulate --grid flags in CLI and apply execution.seed to RNG
  ([`497b5f3`](https://github.com/jacksonpradolima/coleman/commit/497b5f36e290d7bbf4209d965e6c6026cd2017b6))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/6da72e77-e7fa-40f1-b8f0-088c405c5e50

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address all review comments from PR #144
  ([`61276bc`](https://github.com/jacksonpradolima/coleman/commit/61276bc2e19b4cccafd9cafdf1e54d62c28515e0))

- Add seed: int | None = None to ExecutionSpec so sweep seed replication is preserved - Fix
  _set_nested() to validate non-dict intermediates with clear ValueError - Fix resolve_packs() to
  not mutate caller's dict (shallow copy before pop) - Validate equal-length params for zip mode in
  sweep engine (strict=True) - Fix run_id docstring to remove inaccurate "deterministic float
  formatting" claim - Add golden constant (ddd8bbefa143) to test_golden_determinism - Use
  pathlib.Path in run() instead of string concatenation - Guard run_many() against duplicate run_ids
  in parallel mode - Improve test_seed_replication to verify seeds are preserved and affect run_id -
  Add tests: _set_nested non-dict error, zip unequal lengths, resolve_packs immutability

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/082bbecc-0e54-462c-b866-8cbe5da870c0

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Correct spelling 'expend' → 'spent' in runner.py log message
  ([`2881a65`](https://github.com/jacksonpradolima/coleman/commit/2881a6540060219362b2211a4fdc13c408b9a537))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/7ccd72b1-d1d6-49f2-950f-39414e092ac2

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Default ExperimentSpec.policies to ["Random"] only — parametric policies require algorithm config
  from packs
  ([`a5d47b7`](https://github.com/jacksonpradolima/coleman/commit/a5d47b7bd37cb63a1cc8e924aac547000daed9a3))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/445a1d09-34a2-4cbe-8a66-f89a41f8abe1

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Forbid extra fields on fixed specs, resolve packs_dir from config path, seed RNG in test
  ([`4f33366`](https://github.com/jacksonpradolima/coleman/commit/4f333661e8430209a74a7814ea9615e3edd755b5))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/7aba7bf9-535d-409d-86b1-f92b285330b4

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Per-run output paths, validate packs type, use AlgorithmSpec in RunSpec
  ([`1c64a94`](https://github.com/jacksonpradolima/coleman/commit/1c64a94b9a9003163a19ebcaf5138f2ea19abd24))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/cb1ca8a9-e092-4c6f-bd1a-3cc79525ce85

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Update assertion in FRRMABPolicy test to use np.isclose for value estimates
  ([`1dff590`](https://github.com/jacksonpradolima/coleman/commit/1dff590f54ecaf9dbfc491e81eab339efd9f4e47))

- Use pydantic.ValidationError instead of bare Exception in test
  ([`59c800f`](https://github.com/jacksonpradolima/coleman/commit/59c800f1c97e8a415aae39747bf26ea3f43a4fe0))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/c8df2a47-27c5-4b1e-9538-917d3e4a900d

- Use tmp_path for safe temp dirs and pytest.approx for float comparisons in spec tests
  ([`4b77a0e`](https://github.com/jacksonpradolima/coleman/commit/4b77a0e6c1e9b43716e47ae16730c6378d0bf6fc))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/5a4067d7-2b86-4213-963c-2f05e23fbbfe

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Chores

- Update uv.lock for version 1.2.0
  ([`233c2fa`](https://github.com/jacksonpradolima/coleman/commit/233c2fa873ee01d1664cd767fd7fb30a020cc3d9))

- **deps**: Bump anchore/sbom-action from 0.23.0 to 0.23.1 in /.github/workflows
  ([`ec724e3`](https://github.com/jacksonpradolima/coleman/commit/ec724e35c39383f52d2c6150b0eb4e0fc5670bc6))

chore(deps): bump anchore/sbom-action from 0.23.0 to 0.23.1 in /.github/workflows

- **deps**: Bump anchore/sbom-action in /.github/workflows
  ([`4167b57`](https://github.com/jacksonpradolima/coleman/commit/4167b57e52d4fcfb58670067375abd19747f0764))

Bumps [anchore/sbom-action](https://github.com/anchore/sbom-action) from 0.23.0 to 0.23.1. -
  [Release notes](https://github.com/anchore/sbom-action/releases) -
  [Changelog](https://github.com/anchore/sbom-action/blob/main/RELEASE.md) -
  [Commits](https://github.com/anchore/sbom-action/compare/17ae1740179002c89186b61233e0f892c3118b11...57aae528053a48a3f6235f2d9461b05fbcb7366d)

--- updated-dependencies: - dependency-name: anchore/sbom-action dependency-version: 0.23.1

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump codecov/codecov-action from 5.5.2 to 5.5.3 in /.github/workflows
  ([`3c487ae`](https://github.com/jacksonpradolima/coleman/commit/3c487ae4f78f96a893434a704bd3b7048090b9ec))

chore(deps): bump codecov/codecov-action from 5.5.2 to 5.5.3 in /.github/workflows

- **deps**: Bump codecov/codecov-action in /.github/workflows
  ([`83300c0`](https://github.com/jacksonpradolima/coleman/commit/83300c051ba601dd8c0480d2f2ada1d263fc6513))

Bumps [codecov/codecov-action](https://github.com/codecov/codecov-action) from 5.5.2 to 5.5.3. -
  [Release notes](https://github.com/codecov/codecov-action/releases) -
  [Changelog](https://github.com/codecov/codecov-action/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/codecov/codecov-action/compare/671740ac38dd9b0130fbe1cec585b89eea48d3de...1af58845a975a7985b0beb0cbe6fbbb71a41dbad)

--- updated-dependencies: - dependency-name: codecov/codecov-action dependency-version: 5.5.3

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump duckdb from 1.4.4 to 1.5.0
  ([`9ff83a5`](https://github.com/jacksonpradolima/coleman/commit/9ff83a5b93c068bd150eca056098664f79091ed3))

chore(deps): bump duckdb from 1.4.4 to 1.5.0

- **deps**: Bump duckdb from 1.4.4 to 1.5.0
  ([`dd5e313`](https://github.com/jacksonpradolima/coleman/commit/dd5e313ba183ce247bd7fa0c103acde3c2986786))

Bumps [duckdb](https://github.com/duckdb/duckdb-python) from 1.4.4 to 1.5.0. - [Release
  notes](https://github.com/duckdb/duckdb-python/releases) -
  [Commits](https://github.com/duckdb/duckdb-python/compare/v1.4.4...v1.5.0)

--- updated-dependencies: - dependency-name: duckdb dependency-version: 1.5.0

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump numpy from 2.4.2 to 2.4.3
  ([`c8302d2`](https://github.com/jacksonpradolima/coleman/commit/c8302d21cb32383eb9672805008234e8aa11344a))

chore(deps): bump numpy from 2.4.2 to 2.4.3

- **deps**: Bump numpy from 2.4.2 to 2.4.3
  ([`d8cfb3f`](https://github.com/jacksonpradolima/coleman/commit/d8cfb3fba0977f8cb3b01ee82a0839d7bfcc88ef))

Bumps [numpy](https://github.com/numpy/numpy) from 2.4.2 to 2.4.3. - [Release
  notes](https://github.com/numpy/numpy/releases) -
  [Changelog](https://github.com/numpy/numpy/blob/main/doc/RELEASE_WALKTHROUGH.rst) -
  [Commits](https://github.com/numpy/numpy/compare/v2.4.2...v2.4.3)

--- updated-dependencies: - dependency-name: numpy dependency-version: 2.4.3

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump polars from 1.38.1 to 1.39.0
  ([`bada853`](https://github.com/jacksonpradolima/coleman/commit/bada853bba78957f0b4485f5e3931d98b560e79b))

chore(deps): bump polars from 1.38.1 to 1.39.0

- **deps**: Bump polars from 1.38.1 to 1.39.0
  ([`916d8a9`](https://github.com/jacksonpradolima/coleman/commit/916d8a92071378e2825c1efcf2ca8eb03e2ecf59))

Bumps [polars](https://github.com/pola-rs/polars) from 1.38.1 to 1.39.0. - [Release
  notes](https://github.com/pola-rs/polars/releases) -
  [Commits](https://github.com/pola-rs/polars/compare/py-1.38.1...py-1.39.0)

--- updated-dependencies: - dependency-name: polars dependency-version: 1.39.0

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump ruff from 0.15.5 to 0.15.6
  ([`8535f67`](https://github.com/jacksonpradolima/coleman/commit/8535f6774ded6e2e7bb81377e724fc06d8d9000f))

chore(deps): bump ruff from 0.15.5 to 0.15.6

- **deps**: Bump ruff from 0.15.5 to 0.15.6
  ([`58f9616`](https://github.com/jacksonpradolima/coleman/commit/58f961624fd82c07e6804e273554ce6cc0613fb3))

Bumps [ruff](https://github.com/astral-sh/ruff) from 0.15.5 to 0.15.6. - [Release
  notes](https://github.com/astral-sh/ruff/releases) -
  [Changelog](https://github.com/astral-sh/ruff/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/astral-sh/ruff/compare/0.15.5...0.15.6)

--- updated-dependencies: - dependency-name: ruff dependency-version: 0.15.6

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump softprops/action-gh-release from 2.5.0 to 2.6.1 in /.github/workflows
  ([`d4b145b`](https://github.com/jacksonpradolima/coleman/commit/d4b145b6a5741d0b65cf594857359e9db442175d))

chore(deps): bump softprops/action-gh-release from 2.5.0 to 2.6.1 in /.github/workflows

- **deps**: Bump softprops/action-gh-release in /.github/workflows
  ([`9dc56e4`](https://github.com/jacksonpradolima/coleman/commit/9dc56e4bac850687fdb91bea3fcd18c8adb901ac))

Bumps [softprops/action-gh-release](https://github.com/softprops/action-gh-release) from 2.5.0 to
  2.6.1. - [Release notes](https://github.com/softprops/action-gh-release/releases) -
  [Changelog](https://github.com/softprops/action-gh-release/blob/master/CHANGELOG.md) -
  [Commits](https://github.com/softprops/action-gh-release/compare/a06a81a03ee405af7f2048a818ed3f03bbf83c7b...153bb8e04406b158c6c84fc1615b65b24149a1fe)

--- updated-dependencies: - dependency-name: softprops/action-gh-release dependency-version: 2.6.1

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump zensical from 0.0.24 to 0.0.26
  ([`b4e0147`](https://github.com/jacksonpradolima/coleman/commit/b4e0147275dd4ad9e5cd7bac84dbb34bdf4922ae))

chore(deps): bump zensical from 0.0.24 to 0.0.26

- **deps**: Bump zensical from 0.0.24 to 0.0.26
  ([`33713f7`](https://github.com/jacksonpradolima/coleman/commit/33713f73c799f75cc168c5a95c34f319a5ffb016))

Bumps [zensical](https://github.com/zensical/zensical) from 0.0.24 to 0.0.26. - [Release
  notes](https://github.com/zensical/zensical/releases) -
  [Commits](https://github.com/zensical/zensical/compare/v0.0.24...v0.0.26)

--- updated-dependencies: - dependency-name: zensical dependency-version: 0.0.26

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

### Documentation

- Remove all legacy/old references from getting-started.md and README.md
  ([`0e0adb4`](https://github.com/jacksonpradolima/coleman/commit/0e0adb4aed95f6f71bc4e94531f30721ea2abaa4))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/0404e59e-1d4d-461c-99cf-e10db20cdf6b

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Update docs/setup, docs/configuration, remove vNext references, add deep configuration guide
  ([`da6853e`](https://github.com/jacksonpradolima/coleman/commit/da6853e7ce1883b0f98274bcbb9237b49e57810c))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/8369f955-a8d7-4629-852b-8e30e20b7681

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Update README, CONTRIBUTING, and SECURITY for vNext experiment system
  ([`9aa5cf9`](https://github.com/jacksonpradolima/coleman/commit/9aa5cf96c366c360ef87e76c9bd168232a09fe08))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/96ddad8f-82c7-4609-9672-ded24d1bdc6a

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Features

- Experiment system — typed RunSpec + config packs + sweep engine + deterministic run_id + CLI (drop
  TOML/CONFIG_FILE/main.py)
  ([`35a0c44`](https://github.com/jacksonpradolima/coleman/commit/35a0c442f82a4391744534ac320130794a5aae5c))

feat!: experiment system — typed RunSpec + config packs + sweep engine + deterministic run_id + CLI
  (drop TOML/CONFIG_FILE/main.py)

- Vnext experiment system — typed RunSpec + config packs + sweep engine + deterministic run_id + CLI
  ([`7fc7565`](https://github.com/jacksonpradolima/coleman/commit/7fc7565cbd10272beb6fb288cebd1f959f1b4348))

- Add coleman/spec/ with Pydantic v2 models (RunSpec, ExecutionSpec, ExperimentSpec, etc.) -
  Implement deterministic run_id via sha256(canonical_json(resolved_spec))[:12] - Create config pack
  system (packs/ layout + resolver with deep merge) - Implement sweep engine (grid/zip modes with
  stable enumeration + seed replication) - Create coleman/api.py with run(), run_many(),
  sweep(), load_spec(), save_resolved() - Add 'coleman' CLI console script (run, sweep, validate
  commands) - Implement provenance tracking (spec.resolved.json + provenance.json per run) - Add 74
  comprehensive tests for spec models, sweep engine, run_id, packs, API, CLI, provenance - Add
  pydantic + pyyaml dependencies - All 275 tests pass (201 existing + 74 new)

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/c8df2a47-27c5-4b1e-9538-917d3e4a900d

### Refactoring

- Extract _dispatch_executions to reduce run_experiment cognitive complexity from 16 to 8
  ([`61c9326`](https://github.com/jacksonpradolima/coleman/commit/61c9326de9a1d07519e327aba1ecc058e2ccd882))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/d0f17fdb-9b22-49bb-9650-e63981608311

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Move numpy import to top of runner.py and strengthen seed tests
  ([`313f43c`](https://github.com/jacksonpradolima/coleman/commit/313f43ce84fc39255a2be19a3b1225c881225633))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/6da72e77-e7fa-40f1-b8f0-088c405c5e50

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Optimize reward calculation and update value estimates in Agent and Reward classes
  ([`e805527`](https://github.com/jacksonpradolima/coleman/commit/e8055276fba9304522f06c4cae44f01684b19beb))

- Remove config.toml, config/config.py, main.py, .env.example; add packs + run.yaml + runner module
  ([`1bf6fe5`](https://github.com/jacksonpradolima/coleman/commit/1bf6fe5a68c7a5801888618a5377bc89e9fec535))

- Remove config.toml — settings migrated to composable YAML packs - Remove config/config.py — no
  longer needed (TOML loading replaced by YAML) - Remove main.py — replaced by coleman CLI
  entry-point + runner module - Remove .env.example — no env vars needed for configuration - Remove
  python-dotenv, toml, tomli dependencies - Add coleman/runner.py — reusable experiment
  orchestration from main.py - Add run.yaml — default pack-based config for easy project
  understanding - Add new packs: execution/default, experiment/alibaba_druid, algorithm/defaults,
  checkpoint/default, hcs/off, contextual/default, telemetry/local - Update make run → coleman run
  --config run.yaml - Update tests/test_main.py → import from coleman.runner - Update all docs,
  README, CONTRIBUTING, DevContainer scripts

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/7ccd72b1-d1d6-49f2-950f-39414e092ac2

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Testing

- Add comment clarifying intentional typo in extra="forbid" test
  ([`50b619d`](https://github.com/jacksonpradolima/coleman/commit/50b619d30b05c927a719925c4cffc2e937dcc2df))

Agent-Logs-Url:
  https://github.com/jacksonpradolima/coleman/sessions/7aba7bf9-535d-409d-86b1-f92b285330b4

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Add regression test for FRRMABPolicy to handle zero decay sum in credit assignment
  ([`eec4d93`](https://github.com/jacksonpradolima/coleman/commit/eec4d932d544b3e015c04eff05273a659d007f54))


## v1.2.0 (2026-03-22)

### Bug Fixes

- Add lock to ResultsWriter start/stop for thread safety
  ([`a88da89`](https://github.com/jacksonpradolima/coleman/commit/a88da891f5440ec8c5259872fd402ddbe9a30bfa))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address 3 Codex P1 review issues — build-1 skip, per-experiment checkpoints, picklable ParquetSink
  ([`7356b24`](https://github.com/jacksonpradolima/coleman/commit/7356b249f5f8db36cf593177b9e7b0bb42caae25))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address 4 copilot-reviewer comments (notebook extras, dynamic paths, thread restart)
  ([`6f69ab6`](https://github.com/jacksonpradolima/coleman/commit/6f69ab6de42074a6a2669c6c8a64ea1b24c76d40))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address 5 PR review comments — return type, docstrings, variant column, overlay cleanup
  ([`8630323`](https://github.com/jacksonpradolima/coleman/commit/8630323d416b3f62918a1a09a9de47ac4f44de5c))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address 8 review comments — variant column in Parquet/ClickHouse, docstring fixes, config default,
  README corrections, DuckDBCatalog tests
  ([`0dd8759`](https://github.com/jacksonpradolima/coleman/commit/0dd8759de1f7f93634f81e93ddebc201481cfb08))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address code review — remove broad grep guard, fix Python version, clarify create vs start timing
  ([`6d80242`](https://github.com/jacksonpradolima/coleman/commit/6d80242c120695783a71e1b3adea3ee76bd344fe))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address SonarCloud issues and add sonar-project.properties
  ([`9362368`](https://github.com/jacksonpradolima/coleman/commit/93623680345e4b823a69f6b72ab001d8381a2f35))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Address SonarCloud issues — reduce cognitive complexity, extract constants, remove redundant
  returns
  ([`a8d3b60`](https://github.com/jacksonpradolima/coleman/commit/a8d3b60e4b947ffbf557897e3ced645997a5852f))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Ensure thread safety in ResultsWriter and update configuration for telemetry and verbosity
  ([`fb0f676`](https://github.com/jacksonpradolima/coleman/commit/fb0f67672b5ea8df3aa790e95fa626d8fc65e29f))

- Flush result buffers before committing checkpoint step
  ([`d7b2a57`](https://github.com/jacksonpradolima/coleman/commit/d7b2a57db720da221fae1a02c0e489b95f79df04))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Improve DevContainer error messages with actionable guidance
  ([`0a75a67`](https://github.com/jacksonpradolima/coleman/commit/0a75a6722e3d18ca74e63256f02032a74e5046f3))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Make DevContainer more robust — remove third-party uv feature, install uv via curl, use uv sync
  for all extras
  ([`c2de7c7`](https://github.com/jacksonpradolima/coleman/commit/c2de7c7254fd839164f3cadccaaae5bfe9c93c93))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Reject unsupported sink values, close ClickHouse client, process-safe Parquet filenames
  ([`361ccc9`](https://github.com/jacksonpradolima/coleman/commit/361ccc98341ece5acc44eda82e905815ee36f55b))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Replace pandas with polars in DuckDBCatalog, fix OTel unit typo, remove pandas direct dep
  ([`ecb0f31`](https://github.com/jacksonpradolima/coleman/commit/ecb0f31c6572bd59af5f766d8a520cade851b6b2))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Chores

- Ignore runs directory
  ([`a03c6b2`](https://github.com/jacksonpradolima/coleman/commit/a03c6b2e656208cdf657b4d892e7d4fb21b50ff3))

- Update uv.lock for version 1.1.0
  ([`3c99b59`](https://github.com/jacksonpradolima/coleman/commit/3c99b59974ef813921a0953c65b6832facb3ce57))

- **deps**: Bump tornado in the uv group across 1 directory
  ([`3133c1b`](https://github.com/jacksonpradolima/coleman/commit/3133c1b181e5cf7184ef1d68c57bca06d5b8635c))

Bumps the uv group with 1 update in the / directory:
  [tornado](https://github.com/tornadoweb/tornado).

Updates `tornado` from 6.5.4 to 6.5.5 -
  [Changelog](https://github.com/tornadoweb/tornado/blob/master/docs/releases.rst) -
  [Commits](https://github.com/tornadoweb/tornado/compare/v6.5.4...v6.5.5)

--- updated-dependencies: - dependency-name: tornado dependency-version: 6.5.5

dependency-type: indirect

dependency-group: uv

...

Signed-off-by: dependabot[bot] <support@github.com>

- **makefile**: Add run command
  ([`756c8a9`](https://github.com/jacksonpradolima/coleman/commit/756c8a92c7e77befe4363642707c81bada447486))

### Documentation

- Add DevContainer usage documentation to README, Getting Started, and CONTRIBUTING
  ([`d97fd83`](https://github.com/jacksonpradolima/coleman/commit/d97fd830452f2ba40698cbbbaef9b7a5ad9b1979))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Enhance getting started guide with interactive workflow examples for DuckDB
  ([`dc05214`](https://github.com/jacksonpradolima/coleman/commit/dc05214423d8d2f29b5ea5b65624fdd375c9b8df))

### Features

- Add Grafana provisioning for dashboards and datasources
  ([`39f21cc`](https://github.com/jacksonpradolima/coleman/commit/39f21cc6f79e1050bc05ec71b65e7630a0a7775f))

- Introduced default.yml for Grafana dashboards provisioning. - Added default.yml for Prometheus
  datasource configuration. - Created prometheus.yml for Prometheus scraping configuration.

refactor: Enhance main execution logic and telemetry tracking

- Refactored main.py to include new data classes for execution plans and environment configurations.
  - Implemented runtime metadata generation for telemetry. - Improved parallel execution handling
  with responsive Ctrl+C support. - Updated agent creation logic to be more modular and
  maintainable.

test: Add tests for telemetry and resource tracking

- Implemented tests for process resource tracking and telemetry functionalities. - Added regression
  tests for policy behavior and scenario provider normalization.

fix: Ensure scenario provider resets between experiments

- Modified environment.run_single to reset scenario state and telemetry lifecycle for independent
  experiments.

- Make DevContainer fully automatic — observability starts on container boot, zero manual steps
  ([`97c16e9`](https://github.com/jacksonpradolima/coleman/commit/97c16e90e96a2070b69b4865db3a118c9d4a2b7a))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Update DevContainer for full dev experience (uv, Docker-in-Docker, port forwarding)
  ([`ebd9149`](https://github.com/jacksonpradolima/coleman/commit/ebd91490e49b6a83046ae4cb905740b8fd67a8b0))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

### Refactoring

- Add ms-vscode.makefile-tools extension to devcontainer configuration
  ([`4a49deb`](https://github.com/jacksonpradolima/coleman/commit/4a49deb02690053ec6d95ec97a9736f5c2da9d4f))

- Enhance telemetry for checkpoint saves; update observability documentation and Grafana dashboard
  ([`98f4976`](https://github.com/jacksonpradolima/coleman/commit/98f4976013fcb5b1d7d655508291468ee4e24e31))

- Replace MonitorCollector with Results + Telemetry + Checkpoints architecture
  ([`88a0059`](https://github.com/jacksonpradolima/coleman/commit/88a005925b2d16360d6ed041de2053ce57ea61a8))

refactor: replace MonitorCollector with Results + Telemetry + Checkpoints architecture

- Telemetry module to use dynamic imports for OpenTelemetry; enhance resource handling in telemetry
  resources; update documentation for clarity on experiment results and observability; add workflow
  notebook for guided analysis; improve test coverage for checkpoint resumption and parallel
  execution plans.
  ([`f3c310c`](https://github.com/jacksonpradolima/coleman/commit/f3c310ca0f897fdb1a7b5d00c03c273ab17da874))

- Update .gitignore to include checkpoints output; enhance Makefile to install telemetry and
  clickhouse extras
  ([`3a8582d`](https://github.com/jacksonpradolima/coleman/commit/3a8582d80acc1cebb864066633e715499f9f77a9))

- Update linting and formatting checks; enhance Makefile and pre-commit configuration; remove
  obsolete workflow notebook; improve test assertions for policy performance
  ([`2cc55f3`](https://github.com/jacksonpradolima/coleman/commit/2cc55f34417af48bd98ab3ca64634331679098b7))


## v1.1.0 (2026-03-09)

### Chores

- Remove mkdocs.yml and add api-reference.md to nav
  ([`2968abe`](https://github.com/jacksonpradolima/coleman/commit/2968abe5ea3f17c20a271e6239ac0481116cfa27))

- Delete mkdocs.yml since zensical.toml is now the authoritative config - Add api-reference.md as
  Overview entry in the API Reference nav section

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Update uv.lock for version 1.0.0
  ([`f730368`](https://github.com/jacksonpradolima/coleman/commit/f730368e1eb5687d9a0b6e94c0f3668fdc8fdf8e))

- **deps**: Bump actions/checkout from 4 to 6 in /.github/workflows
  ([`8e16b07`](https://github.com/jacksonpradolima/coleman/commit/8e16b072fc13db3b30fe441bf38dcfcc965a4682))

chore(deps): bump actions/checkout from 4 to 6 in /.github/workflows

- **deps**: Bump marimo from 0.20.3 to 0.20.4
  ([`de23fb9`](https://github.com/jacksonpradolima/coleman/commit/de23fb982f958cf32ced2a5eb5bb5212137717b8))

chore(deps): bump marimo from 0.20.3 to 0.20.4

- **deps**: Bump marimo from 0.20.3 to 0.20.4
  ([`d881448`](https://github.com/jacksonpradolima/coleman/commit/d8814489828650fd30d6b7c036a0882ec020cc1e))

Bumps [marimo](https://github.com/marimo-team/marimo) from 0.20.3 to 0.20.4. - [Release
  notes](https://github.com/marimo-team/marimo/releases) -
  [Commits](https://github.com/marimo-team/marimo/compare/0.20.3...0.20.4)

--- updated-dependencies: - dependency-name: marimo dependency-version: 0.20.4

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump ruff from 0.15.4 to 0.15.5
  ([`d19498f`](https://github.com/jacksonpradolima/coleman/commit/d19498f6fc6f948639d61554f88db4def2899ef7))

chore(deps): bump ruff from 0.15.4 to 0.15.5

- **deps**: Bump ruff from 0.15.4 to 0.15.5
  ([`3b4a1bc`](https://github.com/jacksonpradolima/coleman/commit/3b4a1bc3f0d38a8779e21ec05b12bf56bd7506cf))

Bumps [ruff](https://github.com/astral-sh/ruff) from 0.15.4 to 0.15.5. - [Release
  notes](https://github.com/astral-sh/ruff/releases) -
  [Changelog](https://github.com/astral-sh/ruff/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/astral-sh/ruff/compare/0.15.4...0.15.5)

--- updated-dependencies: - dependency-name: ruff dependency-version: 0.15.5

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump sigstore/gh-action-sigstore-python
  ([`c5e36a4`](https://github.com/jacksonpradolima/coleman/commit/c5e36a41316f7a379d100c03df4cdb671948e180))

Bumps [sigstore/gh-action-sigstore-python](https://github.com/sigstore/gh-action-sigstore-python)
  from 3.0.0 to 3.2.0. - [Release
  notes](https://github.com/sigstore/gh-action-sigstore-python/releases) -
  [Changelog](https://github.com/sigstore/gh-action-sigstore-python/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/sigstore/gh-action-sigstore-python/compare/f514d46b907ebcd5bedc05145c03b69c1edd8b46...a5caf349bc536fbef3668a10ed7f5cd309a4b53d)

--- updated-dependencies: - dependency-name: sigstore/gh-action-sigstore-python dependency-version:
  3.2.0

dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump sigstore/gh-action-sigstore-python from 3.0.0 to 3.2.0 in /.github/workflows
  ([`fa36c28`](https://github.com/jacksonpradolima/coleman/commit/fa36c289a9d6c8ecfe39515f86969a16a94fd52a))

chore(deps): bump sigstore/gh-action-sigstore-python from 3.0.0 to 3.2.0 in /.github/workflows

- **deps**: Bump ty from 0.0.20 to 0.0.21
  ([`ae9cd9b`](https://github.com/jacksonpradolima/coleman/commit/ae9cd9be3d1e2e2835f101cd048351452797abee))

chore(deps): bump ty from 0.0.20 to 0.0.21

- **deps**: Bump ty from 0.0.20 to 0.0.21
  ([`f067251`](https://github.com/jacksonpradolima/coleman/commit/f0672512164c4cc1b5258541c60578f21e74a8d2))

Bumps [ty](https://github.com/astral-sh/ty) from 0.0.20 to 0.0.21. - [Release
  notes](https://github.com/astral-sh/ty/releases) -
  [Changelog](https://github.com/astral-sh/ty/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/astral-sh/ty/compare/0.0.20...0.0.21)

--- updated-dependencies: - dependency-name: ty dependency-version: 0.0.21

dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

### Features

- Add zensical.toml configuration for project documentation
  ([`b2114e8`](https://github.com/jacksonpradolima/coleman/commit/b2114e887ae589bc6b4aebe6cc0c0ecf9738e66c))

feat: add zensical.toml configuration for project documentation


## v1.0.0 (2026-03-08)

### Bug Fixes

- Address review - semantic_release branch, vd_a error msg, test tolist, vd_a_df docstring, Makefile
  i18n
  ([`0307188`](https://github.com/jacksonpradolima/coleman/commit/030718803b4d477e5c18d7a9fbde1b967c24db08))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Raise NotImplementedError instead of returning it, use ruff format in Makefile, reformat test
  files
  ([`a49b8c8`](https://github.com/jacksonpradolima/coleman/commit/a49b8c8d87ec972b0eb9c79a3ec8da1fa9422d78))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Resolve all ruff lint errors, add py.typed, restore schema notes, generate uv.lock
  ([`fbea69d`](https://github.com/jacksonpradolima/coleman/commit/fbea69d643b31880ddc32208b31e8d87ee21e56a))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Typing
  ([`54901a6`](https://github.com/jacksonpradolima/coleman/commit/54901a691981ed69567f3c28fa2243b5eb95c72e))

- Use main branch in CI workflows, update docstrings, add encoding, honor symbols param
  ([`bd1a887`](https://github.com/jacksonpradolima/coleman/commit/bd1a887010b2e47385fa8bf792792b09dce3b3ed))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Use make commands in devcontainer, add pytest/ruff settings, migrate to Zensical
  ([`47252c0`](https://github.com/jacksonpradolima/coleman/commit/47252c0d597ee4356772ba8750812826fb7434b8))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- **monitor**: Handle edge cases for empty or invalid batch_df in `collect_from_temp`
  ([`bdd9e1d`](https://github.com/jacksonpradolima/coleman/commit/bdd9e1d96c84fe808e12b7d86fef86d0deb2f774))

- Updated `collect_from_temp` to ensure that only valid non-empty dataframes are concatenated to
  `df`. - Added a condition to check that `batch_df` is not entirely `NaN` before appending. -
  Ensured `temp_rows` is always cleared, regardless of the state of `batch_df`. - Improved
  robustness of the `collect_from_temp` method to prevent unintended behavior.

### Build System

- Update testing dependencies for enhanced mocking capabilities
  ([`8b95ac6`](https://github.com/jacksonpradolima/coleman/commit/8b95ac64b575dc222acd387668e17e2cba882f2f))

Dependencies: + Add pytest-mock==3.14.0 to support mocking in test suites.

### Chores

- Add analysis
  ([`0902a29`](https://github.com/jacksonpradolima/coleman/commit/0902a2913a5c8871fb57b6a8d0dcc560f9d68cc4))

- Add logging to environment.py
  ([`b1a6b9f`](https://github.com/jacksonpradolima/coleman/commit/b1a6b9f55b99a670b887503623db226b200b6a93))

- Add module docstrings
  ([`28465b2`](https://github.com/jacksonpradolima/coleman/commit/28465b2f4ffbee83b21215103973592accd8a061))

- Add scenario name (dataset name) to the monitor.py
  ([`526c821`](https://github.com/jacksonpradolima/coleman/commit/526c821b9fa64f5810c50df3b81f47b5af92a76c))

- Code Refactoring
  ([`318bb1b`](https://github.com/jacksonpradolima/coleman/commit/318bb1bd08afba4619988510b42e2707f59edec9))

- Code Refactoring: Minor Improvements
  ([`43a0108`](https://github.com/jacksonpradolima/coleman/commit/43a01080c02992e21ff0cff262172c2d0c3c6cb8))

- Minor fixes and moved from unittest to pytest
  ([`bec2451`](https://github.com/jacksonpradolima/coleman/commit/bec2451300dd67d39cf1e98a4b6b6eb1864c2414))

- Removed old analysis
  ([`1e16f5a`](https://github.com/jacksonpradolima/coleman/commit/1e16f5adb861812cd43eb18402ee64d7947cc036))

- Suppress warnings and clean up code
  ([`6f5e8b9`](https://github.com/jacksonpradolima/coleman/commit/6f5e8b9014bb4093a45f39053f97369c96bb20b8))

Environment Setup: + Suppress FutureWarning messages globally using `warnings.simplefilter`.

Data Processing: # Update fillna method for clarity in `scenarios.py`.

Testing: - Remove unused import `group` in `test_monitor.py`.

- Update requirements
  ([`1bf574f`](https://github.com/jacksonpradolima/coleman/commit/1bf574f47f091249b39a7a38f9eac77d17840a34))

- **analysis**: Remove unnecessary cell
  ([`b00bf86`](https://github.com/jacksonpradolima/coleman/commit/b00bf862df62b9463940d527d8283576726178f4))

- **config**: Add new examples
  ([`bd975ed`](https://github.com/jacksonpradolima/coleman/commit/bd975edb322da90f2d4587b174e5c210ee0c53cf))

- **deps**: Bump actions/checkout from 3 to 4 in /.github/workflows
  ([`0bd4d69`](https://github.com/jacksonpradolima/coleman/commit/0bd4d6962a2f7ca44354ba03bd683f6f1e444601))

Bumps [actions/checkout](https://github.com/actions/checkout) from 3 to 4. - [Release
  notes](https://github.com/actions/checkout/releases) -
  [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/actions/checkout/compare/v3...v4)

--- updated-dependencies: - dependency-name: actions/checkout dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump actions/checkout from 4 to 6 in /.github/workflows
  ([`24bdbbb`](https://github.com/jacksonpradolima/coleman/commit/24bdbbb7f87d497729a6213c195b5caad401a9e9))

Bumps [actions/checkout](https://github.com/actions/checkout) from 4 to 6. - [Release
  notes](https://github.com/actions/checkout/releases) -
  [Changelog](https://github.com/actions/checkout/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/actions/checkout/compare/v4...v6)

--- updated-dependencies: - dependency-name: actions/checkout dependency-version: '6'

dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump actions/setup-python from 2 to 4 in /.github/workflows
  ([`734dc6b`](https://github.com/jacksonpradolima/coleman/commit/734dc6bca81c7fa8033e6d3d2808e732623b1a8e))

Bumps [actions/setup-python](https://github.com/actions/setup-python) from 2 to 4. - [Release
  notes](https://github.com/actions/setup-python/releases) -
  [Commits](https://github.com/actions/setup-python/compare/v2...v4)

--- updated-dependencies: - dependency-name: actions/setup-python dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump actions/setup-python from 4 to 5 in /.github/workflows
  ([`b2557f5`](https://github.com/jacksonpradolima/coleman/commit/b2557f58ef3495bc5b06d0965cb50393e0c47e83))

Bumps [actions/setup-python](https://github.com/actions/setup-python) from 4 to 5. - [Release
  notes](https://github.com/actions/setup-python/releases) -
  [Commits](https://github.com/actions/setup-python/compare/v4...v5)

--- updated-dependencies: - dependency-name: actions/setup-python dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump actions/setup-python from 5 to 6 in /.github/workflows
  ([`01f2a25`](https://github.com/jacksonpradolima/coleman/commit/01f2a2528c5ee21afcd433812e5a973f5423594f))

Bumps [actions/setup-python](https://github.com/actions/setup-python) from 5 to 6. - [Release
  notes](https://github.com/actions/setup-python/releases) -
  [Commits](https://github.com/actions/setup-python/compare/v5...v6)

--- updated-dependencies: - dependency-name: actions/setup-python dependency-version: '6'

dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump duckdb from 0.10.0 to 0.10.1
  ([`6a9cc75`](https://github.com/jacksonpradolima/coleman/commit/6a9cc75752f9d1d471ba6d496bc4b80b7fe3538f))

Bumps [duckdb](https://github.com/duckdb/duckdb) from 0.10.0 to 0.10.1. - [Release
  notes](https://github.com/duckdb/duckdb/releases) -
  [Changelog](https://github.com/duckdb/duckdb/blob/main/tools/release-pip.py) -
  [Commits](https://github.com/duckdb/duckdb/compare/v0.10.0...v0.10.1)

--- updated-dependencies: - dependency-name: duckdb dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump duckdb from 0.10.1 to 0.10.2
  ([`6cb58af`](https://github.com/jacksonpradolima/coleman/commit/6cb58af345ed8648d4df62f06fa4fae936e78106))

Bumps [duckdb](https://github.com/duckdb/duckdb) from 0.10.1 to 0.10.2. - [Release
  notes](https://github.com/duckdb/duckdb/releases) -
  [Changelog](https://github.com/duckdb/duckdb/blob/main/tools/release-pip.py) -
  [Commits](https://github.com/duckdb/duckdb/compare/v0.10.1...v0.10.2)

--- updated-dependencies: - dependency-name: duckdb dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump duckdb from 0.10.2 to 1.1.3
  ([`ca2b060`](https://github.com/jacksonpradolima/coleman/commit/ca2b060acd5d9337d5ec6c30d9b08132513c4739))

Bumps [duckdb](https://github.com/duckdb/duckdb) from 0.10.2 to 1.1.3. - [Release
  notes](https://github.com/duckdb/duckdb/releases) -
  [Changelog](https://github.com/duckdb/duckdb/blob/main/tools/release-pip.py) -
  [Commits](https://github.com/duckdb/duckdb/compare/v0.10.2...v1.1.3)

--- updated-dependencies: - dependency-name: duckdb dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump numpy from 1.25.2 to 1.26.4
  ([`5c0b64f`](https://github.com/jacksonpradolima/coleman/commit/5c0b64f1d74c83c041ec1e8f81ad0c4b127500e4))

Bumps [numpy](https://github.com/numpy/numpy) from 1.25.2 to 1.26.4. - [Release
  notes](https://github.com/numpy/numpy/releases) -
  [Changelog](https://github.com/numpy/numpy/blob/main/doc/RELEASE_WALKTHROUGH.rst) -
  [Commits](https://github.com/numpy/numpy/compare/v1.25.2...v1.26.4)

--- updated-dependencies: - dependency-name: numpy dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump numpy from 1.26.4 to 2.0.1
  ([`7454fb0`](https://github.com/jacksonpradolima/coleman/commit/7454fb0ee8854183705774986ba7ce98e5c86186))

Bumps [numpy](https://github.com/numpy/numpy) from 1.26.4 to 2.0.1. - [Release
  notes](https://github.com/numpy/numpy/releases) -
  [Changelog](https://github.com/numpy/numpy/blob/main/doc/RELEASE_WALKTHROUGH.rst) -
  [Commits](https://github.com/numpy/numpy/compare/v1.26.4...v2.0.1)

--- updated-dependencies: - dependency-name: numpy dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pandas from 2.1.0 to 2.2.0
  ([`5576969`](https://github.com/jacksonpradolima/coleman/commit/557696984d1c05f26efd9f9216f16402cb6dbc1d))

Bumps [pandas](https://github.com/pandas-dev/pandas) from 2.1.0 to 2.2.0. - [Release
  notes](https://github.com/pandas-dev/pandas/releases) -
  [Commits](https://github.com/pandas-dev/pandas/compare/v2.1.0...v2.2.0)

--- updated-dependencies: - dependency-name: pandas dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pandas from 2.2.0 to 2.2.1
  ([`d4dabce`](https://github.com/jacksonpradolima/coleman/commit/d4dabcef4275b22381843cce4efb0ebd5ebc11ec))

Bumps [pandas](https://github.com/pandas-dev/pandas) from 2.2.0 to 2.2.1. - [Release
  notes](https://github.com/pandas-dev/pandas/releases) -
  [Commits](https://github.com/pandas-dev/pandas/compare/v2.2.0...v2.2.1)

--- updated-dependencies: - dependency-name: pandas dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pandas from 2.2.1 to 2.2.2
  ([`04cbbe2`](https://github.com/jacksonpradolima/coleman/commit/04cbbe237c457b447148038d11ea9f5e24144ad2))

Bumps [pandas](https://github.com/pandas-dev/pandas) from 2.2.1 to 2.2.2. - [Release
  notes](https://github.com/pandas-dev/pandas/releases) -
  [Commits](https://github.com/pandas-dev/pandas/compare/v2.2.1...v2.2.2)

--- updated-dependencies: - dependency-name: pandas dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pandas from 2.2.2 to 2.2.3
  ([`0bb2055`](https://github.com/jacksonpradolima/coleman/commit/0bb2055718038a2c464eaff11373241e78638250))

Bumps [pandas](https://github.com/pandas-dev/pandas) from 2.2.2 to 2.2.3. - [Release
  notes](https://github.com/pandas-dev/pandas/releases) -
  [Commits](https://github.com/pandas-dev/pandas/compare/v2.2.2...v2.2.3)

--- updated-dependencies: - dependency-name: pandas dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pyarrow from 15.0.1 to 15.0.2
  ([`ed90e7f`](https://github.com/jacksonpradolima/coleman/commit/ed90e7fac77200e77bca8d21fcd4d0d0dd832a69))

Bumps [pyarrow](https://github.com/apache/arrow) from 15.0.1 to 15.0.2. -
  [Commits](https://github.com/apache/arrow/compare/go/v15.0.1...go/v15.0.2)

--- updated-dependencies: - dependency-name: pyarrow dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pylint from 2.17.5 to 3.0.3
  ([`ded60a0`](https://github.com/jacksonpradolima/coleman/commit/ded60a02747b461e73ba30073c887a6364af120e))

Bumps [pylint](https://github.com/pylint-dev/pylint) from 2.17.5 to 3.0.3. - [Release
  notes](https://github.com/pylint-dev/pylint/releases) -
  [Commits](https://github.com/pylint-dev/pylint/compare/v2.17.5...v3.0.3)

--- updated-dependencies: - dependency-name: pylint dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pylint from 3.0.3 to 3.1.0
  ([`846abb6`](https://github.com/jacksonpradolima/coleman/commit/846abb6b611d21eaced34cb800796525a7310dd1))

Bumps [pylint](https://github.com/pylint-dev/pylint) from 3.0.3 to 3.1.0. - [Release
  notes](https://github.com/pylint-dev/pylint/releases) -
  [Commits](https://github.com/pylint-dev/pylint/compare/v3.0.3...v3.1.0)

--- updated-dependencies: - dependency-name: pylint dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pylint from 3.1.0 to 3.2.6
  ([`836cec1`](https://github.com/jacksonpradolima/coleman/commit/836cec1a7779c531d0ccc1c81308562d0e55f7de))

Bumps [pylint](https://github.com/pylint-dev/pylint) from 3.1.0 to 3.2.6. - [Release
  notes](https://github.com/pylint-dev/pylint/releases) -
  [Commits](https://github.com/pylint-dev/pylint/compare/v3.1.0...v3.2.6)

--- updated-dependencies: - dependency-name: pylint dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pylint from 3.2.6 to 3.3.3
  ([`17d039d`](https://github.com/jacksonpradolima/coleman/commit/17d039d6bf7e7835278bf986709392fb62ca1c1a))

Bumps [pylint](https://github.com/pylint-dev/pylint) from 3.2.6 to 3.3.3. - [Release
  notes](https://github.com/pylint-dev/pylint/releases) -
  [Commits](https://github.com/pylint-dev/pylint/compare/v3.2.6...v3.3.3)

--- updated-dependencies: - dependency-name: pylint dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump pylint from 3.3.3 to 3.3.4
  ([`53e4e98`](https://github.com/jacksonpradolima/coleman/commit/53e4e9841f445be8f1b49e9f2e12129227c127e9))

Bumps [pylint](https://github.com/pylint-dev/pylint) from 3.3.3 to 3.3.4. - [Release
  notes](https://github.com/pylint-dev/pylint/releases) -
  [Commits](https://github.com/pylint-dev/pylint/compare/v3.3.3...v3.3.4)

--- updated-dependencies: - dependency-name: pylint dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump python-dotenv from 1.0.0 to 1.0.1
  ([`0bb9ee6`](https://github.com/jacksonpradolima/coleman/commit/0bb9ee6793868a7b045f419d1fc73197c2777230))

Bumps [python-dotenv](https://github.com/theskumar/python-dotenv) from 1.0.0 to 1.0.1. - [Release
  notes](https://github.com/theskumar/python-dotenv/releases) -
  [Changelog](https://github.com/theskumar/python-dotenv/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/theskumar/python-dotenv/compare/v1.0.0...v1.0.1)

--- updated-dependencies: - dependency-name: python-dotenv dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump scikit-posthocs from 0.9.0 to 0.11.2
  ([`69216ab`](https://github.com/jacksonpradolima/coleman/commit/69216abe8c05e08ee5a9f2d7aaa41b4d41561361))

Bumps [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) from 0.9.0 to 0.11.2. -
  [Release notes](https://github.com/maximtrp/scikit-posthocs/releases) -
  [Commits](https://github.com/maximtrp/scikit-posthocs/compare/v0.9.0...v0.11.2)

--- updated-dependencies: - dependency-name: scikit-posthocs dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump scipy from 1.10.1 to 1.13.0
  ([`54acf40`](https://github.com/jacksonpradolima/coleman/commit/54acf40c0a858ff03d7fe8c7018f75f4796d05cc))

Bumps [scipy](https://github.com/scipy/scipy) from 1.10.1 to 1.13.0. - [Release
  notes](https://github.com/scipy/scipy/releases) -
  [Commits](https://github.com/scipy/scipy/compare/v1.10.1...v1.13.0)

--- updated-dependencies: - dependency-name: scipy dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump scipy from 1.13.0 to 1.14.0
  ([`4ba8165`](https://github.com/jacksonpradolima/coleman/commit/4ba8165af7281d0414527a7fabe7bb43c71dc3ec))

Bumps [scipy](https://github.com/scipy/scipy) from 1.13.0 to 1.14.0. - [Release
  notes](https://github.com/scipy/scipy/releases) -
  [Commits](https://github.com/scipy/scipy/compare/v1.13.0...v1.14.0)

--- updated-dependencies: - dependency-name: scipy dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump scipy from 1.14.0 to 1.14.1
  ([`e0c3edf`](https://github.com/jacksonpradolima/coleman/commit/e0c3edfb17357cabd3c8a23f9a875888324a4ecf))

Bumps [scipy](https://github.com/scipy/scipy) from 1.14.0 to 1.14.1. - [Release
  notes](https://github.com/scipy/scipy/releases) -
  [Commits](https://github.com/scipy/scipy/compare/v1.14.0...v1.14.1)

--- updated-dependencies: - dependency-name: scipy dependency-type: direct:production

update-type: version-update:semver-patch

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump tj-actions/changed-files in /.github/workflows
  ([`a85e83d`](https://github.com/jacksonpradolima/coleman/commit/a85e83d8cc7f0f20ab2e6ecabdc13080e0a59491))

Bumps [tj-actions/changed-files](https://github.com/tj-actions/changed-files) from 46 to 47. -
  [Release notes](https://github.com/tj-actions/changed-files/releases) -
  [Changelog](https://github.com/tj-actions/changed-files/blob/main/HISTORY.md) -
  [Commits](https://github.com/tj-actions/changed-files/compare/v46...v47)

--- updated-dependencies: - dependency-name: tj-actions/changed-files dependency-version: '47'

dependency-type: direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump tj-actions/changed-files in /.github/workflows
  ([`0c0442d`](https://github.com/jacksonpradolima/coleman/commit/0c0442db8e350487b62e1de6bab4a82f0b1a1b69))

Bumps [tj-actions/changed-files](https://github.com/tj-actions/changed-files) from 45 to 46. -
  [Release notes](https://github.com/tj-actions/changed-files/releases) -
  [Changelog](https://github.com/tj-actions/changed-files/blob/main/HISTORY.md) -
  [Commits](https://github.com/tj-actions/changed-files/compare/v45...v46)

--- updated-dependencies: - dependency-name: tj-actions/changed-files dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump tj-actions/changed-files in /.github/workflows
  ([`c4de901`](https://github.com/jacksonpradolima/coleman/commit/c4de901e94ed633bfe2d7d6c3fa29be4abc807b5))

Bumps [tj-actions/changed-files](https://github.com/tj-actions/changed-files) from 44 to 45. -
  [Release notes](https://github.com/tj-actions/changed-files/releases) -
  [Changelog](https://github.com/tj-actions/changed-files/blob/main/HISTORY.md) -
  [Commits](https://github.com/tj-actions/changed-files/compare/v44...v45)

--- updated-dependencies: - dependency-name: tj-actions/changed-files dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump tj-actions/changed-files in /.github/workflows
  ([`f89ab47`](https://github.com/jacksonpradolima/coleman/commit/f89ab475fd6b8a0610ff582567ab671a616007bd))

Bumps [tj-actions/changed-files](https://github.com/tj-actions/changed-files) from 43 to 44. -
  [Release notes](https://github.com/tj-actions/changed-files/releases) -
  [Changelog](https://github.com/tj-actions/changed-files/blob/main/HISTORY.md) -
  [Commits](https://github.com/tj-actions/changed-files/compare/v43...v44)

--- updated-dependencies: - dependency-name: tj-actions/changed-files dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump tj-actions/changed-files in /.github/workflows
  ([`16825e6`](https://github.com/jacksonpradolima/coleman/commit/16825e627bb58c7ff8812db97336a5f19b9fedea))

Bumps [tj-actions/changed-files](https://github.com/tj-actions/changed-files) from 42 to 43. -
  [Release notes](https://github.com/tj-actions/changed-files/releases) -
  [Changelog](https://github.com/tj-actions/changed-files/blob/main/HISTORY.md) -
  [Commits](https://github.com/tj-actions/changed-files/compare/v42...v43)

--- updated-dependencies: - dependency-name: tj-actions/changed-files dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump tj-actions/changed-files in /.github/workflows
  ([`2086173`](https://github.com/jacksonpradolima/coleman/commit/2086173766a228bb4a91dd9123f70c22a2c329b8))

Bumps [tj-actions/changed-files](https://github.com/tj-actions/changed-files) from 40 to 42. -
  [Release notes](https://github.com/tj-actions/changed-files/releases) -
  [Changelog](https://github.com/tj-actions/changed-files/blob/main/HISTORY.md) -
  [Commits](https://github.com/tj-actions/changed-files/compare/v40...v42)

--- updated-dependencies: - dependency-name: tj-actions/changed-files dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump tj-actions/changed-files in /.github/workflows
  ([`92d69c0`](https://github.com/jacksonpradolima/coleman/commit/92d69c0087254620ca172c1c025cf8c8e211bdb6))

Bumps [tj-actions/changed-files](https://github.com/tj-actions/changed-files) from 39 to 40. -
  [Release notes](https://github.com/tj-actions/changed-files/releases) -
  [Changelog](https://github.com/tj-actions/changed-files/blob/main/HISTORY.md) -
  [Commits](https://github.com/tj-actions/changed-files/compare/v39...v40)

--- updated-dependencies: - dependency-name: tj-actions/changed-files dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump tj-actions/changed-files in /.github/workflows
  ([`19abf25`](https://github.com/jacksonpradolima/coleman/commit/19abf25d1e2ed4efcda3f3c2c541fe090b895fa7))

Bumps [tj-actions/changed-files](https://github.com/tj-actions/changed-files) from 35 to 39. -
  [Release notes](https://github.com/tj-actions/changed-files/releases) -
  [Changelog](https://github.com/tj-actions/changed-files/blob/main/HISTORY.md) -
  [Commits](https://github.com/tj-actions/changed-files/compare/v35...v39)

--- updated-dependencies: - dependency-name: tj-actions/changed-files dependency-type:
  direct:production

update-type: version-update:semver-major

...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump tomli from 2.0.1 to 2.2.1
  ([`1e4a806`](https://github.com/jacksonpradolima/coleman/commit/1e4a80606357fbc7811b6229a3ad4d5e7f784057))

Bumps [tomli](https://github.com/hukkin/tomli) from 2.0.1 to 2.2.1. -
  [Changelog](https://github.com/hukkin/tomli/blob/master/CHANGELOG.md) -
  [Commits](https://github.com/hukkin/tomli/compare/2.0.1...2.2.1)

--- updated-dependencies: - dependency-name: tomli dependency-type: direct:production

update-type: version-update:semver-minor

...

Signed-off-by: dependabot[bot] <support@github.com>

- **examples**: Add alibaba@druid as new example
  ([`be3fc50`](https://github.com/jacksonpradolima/coleman/commit/be3fc50979c3450a738a2750e1a77dbbbceef9ce))

- **main**: Add DuckDB and Logging
  ([`a145e6e`](https://github.com/jacksonpradolima/coleman/commit/a145e6e932aca3a031ec23957931561711c9950d))

- **main**: Add missing variable
  ([`26029ca`](https://github.com/jacksonpradolima/coleman/commit/26029caf78e8db5410788eea0e13c3cc930b5527))

- **main**: Undo comments.
  ([`c4e1ae4`](https://github.com/jacksonpradolima/coleman/commit/c4e1ae4e05889c29d8ace6920f1ca002c803d4c8))

- **main,environment**: Remove spaces
  ([`255276b`](https://github.com/jacksonpradolima/coleman/commit/255276ba3be8cd18a6adce11cffde355651df178))

- **main,readme**: Minor improvements
  ([`a2a4e9d`](https://github.com/jacksonpradolima/coleman/commit/a2a4e9da15cc909905dfacf6a8190e575981751b))

- **monitor_params**: Add class docstring
  ([`ffc5670`](https://github.com/jacksonpradolima/coleman/commit/ffc5670d3e29ec1175001a5909e1312bca174c40))

- **readme**: Fix typo
  ([`591dc56`](https://github.com/jacksonpradolima/coleman/commit/591dc568167e43720e71eacb642234aa22eb9ca5))

### Documentation

- Update mermaid diagram to requested flow wording
  ([`6512846`](https://github.com/jacksonpradolima/coleman/commit/6512846d4f2639a15183add1c801b85c9873faed))

- Update notebook reference in README to point to new location
  ([`1ff8015`](https://github.com/jacksonpradolima/coleman/commit/1ff8015bf837c89b7a24f9e23094e1d881ef1288))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Update README for UV/pyproject.toml migration and add Development section
  ([`f259aed`](https://github.com/jacksonpradolima/coleman/commit/f259aedbb5614790190df574671cb511d638b75f))

- Update Python badge from 3.11.4+ to 3.14+ - Replace Installation section:
  pyenv/pip/requirements.txt → UV + uv sync - Add Development section with Makefile commands table -
  Update run command to use 'uv run python main.py' - Add links to CONTRIBUTING.md and SECURITY.md
  in Contributors section - Fix stale TOC link for Installation

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>

### Features

- Add CODEOWNERS, FUNDING, and labeler configuration files
  ([`d495232`](https://github.com/jacksonpradolima/coleman/commit/d4952325303931b6c4bde1e646bd62a48d3f3d85))

- Add issue templates for bug reports and feature requests
  ([`ccbdb9e`](https://github.com/jacksonpradolima/coleman/commit/ccbdb9ebc67a0c006dbec3bcb69f2645e2e32d5a))

- Add testing for environment functionality and improve maintainability
  ([`c01cd41`](https://github.com/jacksonpradolima/coleman/commit/c01cd41fc989595cf395d79c8be5fba5b7dbc164))

Testing: + Add comprehensive unit tests for the Environment class, including core methods. + Verify
  scenario handling, prioritization, and file operations (e.g., save/load backups). + Ensure
  exception handling tests for critical methods like save_experiment and load_experiment. +
  Introduce mocks for agents, scenarios, metrics, and external resources.

Environment: + Add logging support for better issue tracking and debugging. + Improve method
  consistency by renaming parameter `vsc` to `virtual_scenario` for clarity. + Enhance exception
  handling in save_experiment to capture and log errors. # Fix potential issues with file handling
  when saving/loading backups. # Improve inline documentation for better readability and
  scalability.

Refactor: - Remove unused imports and parameters (e.g., DynamicBandit references). # Update method
  definitions and calls for improved readability and alignment with new logging.

Testing Tools: + Add pytest fixture files for mocking agents, metrics, and scenario providers. +
  Mock external resources (e.g., files, databases) to isolate test environments.

- Adicionar extensões do VSCode para suporte a Python e Jupyter
  ([`879505c`](https://github.com/jacksonpradolima/coleman/commit/879505c6118b54b69728feeef08fb90f363f7d85))

- Adicionar workflows para validação de PR, revisão de dependências e geração de SLSA provenance
  ([`cdd1d63`](https://github.com/jacksonpradolima/coleman/commit/cdd1d630d03964823263bfc5341f2f3d3cca2aa0))

- **agent**: Optimize action management and reduce execution time by 69%
  ([`9eee1bc`](https://github.com/jacksonpradolima/coleman/commit/9eee1bcc77239d9d7422e49e23d4da00ec011f77))

- Refactored the `add_action` method to eliminate redundancy by checking for the existence of
  actions before adding them. - Streamlined the `update_actions` method to handle obsolete and new
  actions efficiently, reducing unnecessary DataFrame manipulations. - Optimized random selection of
  actions in the `choose` method by using `random.sample`. - Enhanced `update_action_attempts` to
  map action indices efficiently with a dictionary for weights, minimizing computational overhead. -
  Overall performance improvement: Reduced experiment execution time from 92s to 28s (~69%
  reduction).

- **bandit**: Improve performance of priority updates and arm addition
  ([`343f0b8`](https://github.com/jacksonpradolima/coleman/commit/343f0b84b1f654290db66130bf809888eaf5a1d8))

- Replaced lambda-based priority updates with `numpy.vectorize` for better performance in
  `update_priority`. - Added a check to prevent unnecessary `pd.concat` calls in `add_arms` if the
  arm list is empty. - Introduced validation for empty actions in `EvaluationMetricBandit.pull`.

- **bandit**: Minor improvements to speed up bandit
  ([`f8cd99c`](https://github.com/jacksonpradolima/coleman/commit/f8cd99c4c373558ecfcc3455f6b79971f9b00351))

- **codecov**: Add codecov to github workflow
  ([`20d2257`](https://github.com/jacksonpradolima/coleman/commit/20d2257aafca4c8fe14379e448853f99d1f62857))

- **codecov**: Use requirements instead of specific packages
  ([`89af3fc`](https://github.com/jacksonpradolima/coleman/commit/89af3fcd7b48ed676dbadccfe35b597f61c62931))

- **gh-action**: Add test results to codecov
  ([`7a4fa9b`](https://github.com/jacksonpradolima/coleman/commit/7a4fa9b0768f4fa4aa3e6728b5fd3c155bca5b5e))

- **monitor**: Optimize data collection with row batching, boosting performance by 19,000x
  ([`de6f25f`](https://github.com/jacksonpradolima/coleman/commit/de6f25f7f1a4f87328787177fb54dcf39f230940))

- Refactored `MonitorCollector` to use row batching (`temp_rows`) instead of DataFrame operations
  for temporary storage, improving efficiency. - Replaced DataFrame concatenation logic with batched
  appends, minimizing memory overhead and computational costs. - Updated save logic to dynamically
  handle headers based on file existence and state. - Enhanced `collect_from_temp` to ensure
  DataFrame consistency while significantly reducing execution time. - Adjusted performance
  benchmarks to reflect improvements, confirming a remarkable **19,000x speed increase** for
  large-scale experiment data processing.

Other changes: - Updated `pytest` performance tests and configurations. - Adjusted policies and
  configurations in `config.toml` for streamlined execution. - Included new dependencies
  (`pytest-benchmark`, `pytest-cov`) in `requirements.txt`.

Breaking changes: - Removed direct use of temporary DataFrame (`temp_df`), replacing it with a
  row-based batch system (`temp_rows`).

Benchmarks: - Small-scale tests (1,000 records): ~4ms (down from ~800ms). - Medium-scale tests
  (10,000 records): ~45ms (down from ~8,000ms). - Large-scale tests (100,000 records): ~579ms (down
  from ~83,000ms).

- **readme**: Add codecov badge
  ([`d484eb2`](https://github.com/jacksonpradolima/coleman/commit/d484eb234b2377734025a65d93b71d9e897b95a6))

- **reward**: Optimize RNFailReward and enhance tests
  ([`88842a8`](https://github.com/jacksonpradolima/coleman/commit/88842a8f2da4dfb85d26896e41c18b56f4ce75fe))

- Improved the `RNFailReward` evaluation logic by directly iterating over detection ranks, reducing
  memory usage. - Updated `RNFailReward` to assign rewards directly, improving performance for large
  datasets. - Added performance tests for `TimeRankReward` and `RNFailReward` to ensure scalability
  and efficiency. - Enhanced unit test coverage for edge cases and performance scenarios. - Adjusted
  benchmark settings to validate the performance improvements.

- **statistics**: Add test cases and restructure project layout
  ([`8cb0146`](https://github.com/jacksonpradolima/coleman/commit/8cb01467c5c9f79756e44bf35093e350079ce18d))

- Added unit tests for Vargha and Delaney's A index, including edge cases and different effect
  sizes. - Renamed and relocated `evaluation_test.py` and `policy_test.py` to align with the new
  project structure under `tests/`. - Introduced a new test file `test_varga_delaney.py` with
  comprehensive coverage. - Updated requirements with new dependencies: `flake8`, `pytest-cov`,
  `snakeviz`, and upgraded `pyarrow`. - Enhanced `.gitignore` to exclude coverage files (e.g.,
  `htmlcov/`).

- **TimeRankReward**: Enhance evaluate method with normalization logic
  ([`c7a1253`](https://github.com/jacksonpradolima/coleman/commit/c7a125324d72dc7b56f4e316fcfffef3e25eaf24))

- Updated the `evaluate` method in the `TimeRankReward` class to normalize rewards by dividing them
  by the number of failing test cases (`num_failing_tests`). - Added detailed inline comments to
  explain the normalization logic, ensuring clarity and alignment with the TimeRank equation. -
  Enhanced docstring to document the parameters, process, and return value of the `evaluate` method.

### Refactoring

- Add UV pyproject.toml, pre-commit, Makefile, devcontainer, CI workflows, SECURITY, CONTRIBUTING,
  CITATION, CHANGELOG
  ([`0c52c88`](https://github.com/jacksonpradolima/coleman/commit/0c52c88a2f1ace2379fe974d70c0891be04dfcab))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Aprimorar o fluxo de trabalho de lançamento automatizado com melhorias na configuração do Python e
  gerenciamento de versões
  ([`6e62bd3`](https://github.com/jacksonpradolima/coleman/commit/6e62bd3dba42975867a33d58c8b31f4b7618fd87))

- Atualizar configuração do projeto e melhorar a documentação
  ([`4e6f19e`](https://github.com/jacksonpradolima/coleman/commit/4e6f19ef471bb9bb9999bba895becee65da8d7b6))

- Clean up unused code and improve class initialization
  ([`9ccc515`](https://github.com/jacksonpradolima/coleman/commit/9ccc515fc8b85cf7b7acd2dc94439ced49a364c5))

DynamicBandit: - Remove commented-out column conversion logic for clarity. - Simplify class
  initialization by removing redundant steps.

Maintenance: # Streamline `reset` and `add_arms` methods for better readability.

- Improve test consistency and enhance randomization logic
  ([`b7eb93e`](https://github.com/jacksonpradolima/coleman/commit/b7eb93e31168bc920e4825a4ffca8923cebc93b5))

Tests: - Remove commented-out test `test_get_scenario_provider_hcs`. # Update test
  `test_environment` to discard unused unpacked variables in method call. # Replace
  `np.random.rand()` with `np.random.default_rng().random()` for better randomization control. #
  Refactor helper method `simulate_actions` to use snake_case for argument `include_q`. # Adjust
  benchmark tests to align with updated helper function parameter naming.

Core Logic: - Remove unused commented code in `coleman/scenarios.py` to improve code clarity.

- Migrate to UV, pyproject.toml, ruff, Zensical, and modern Python toolchain
  ([`88c55ed`](https://github.com/jacksonpradolima/coleman/commit/88c55ed3e9777e657fff99ae9c6f27d93bce7b05))

refactor!: migrate to UV, pyproject.toml, ruff, Zensical, and modern Python toolchain

- Pandas to Polars migration
  ([`50decab`](https://github.com/jacksonpradolima/coleman/commit/50decab44b08819c5b07e0abb7146cb94bd49463))

refactor: Pandas to Polars migration

- Rename VD_A/VD_A_DF to snake_case, use numpy.random.Generator, rename A_inv to a_inv
  ([`efafbb5`](https://github.com/jacksonpradolima/coleman/commit/efafbb5e63c92e899b23e8562fdb8bb5641acd55))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Replace np.random with rng for random number generation in tests
  ([`3bb15e6`](https://github.com/jacksonpradolima/coleman/commit/3bb15e622a877558110225563b47164a4c4e5603))

Testing: # Update tests to use `rng.integers` instead of `np.random.randint`. # Ensure consistency
  and alignment with the updated random number generation API.

- Simplify vd_a estimate variable naming
  ([`df25313`](https://github.com/jacksonpradolima/coleman/commit/df253130fbd8e60036ccd946ad9d6c9e897526e6))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- Standardize RNG usage and enhance test reproducibility
  ([`290e43e`](https://github.com/jacksonpradolima/coleman/commit/290e43e5e10b81988aadd29b66de5d91e3592f01))

Test Configuration: + Add fixed random seed for reproducibility across all tests. + Initialize
  global RNG instance (`rng`) for consistent sampling.

Test Logic: # Replace direct `np.random.default_rng()` calls with `rng`. # Update test fixtures and
  agents to use standardized random values.

- Streamline scenario management and enhance save functionality
  ([`89a8e49`](https://github.com/jacksonpradolima/coleman/commit/89a8e499db2a3fa12cbf583a15abf06aa5076bf4))

Scenario Management: - Remove `has_variants` method; inline logic to check scenario variants. #
  Update condition to explicitly check `VirtualHCSScenario` type and its variants.

Save Functionality: + Add `interval` parameter to `save_periodically` for configurable save
  frequency. # Refactor save logic to use dynamic interval instead of a fixed value.

Dependencies: - Remove unused imports from `scenarios.py` (e.g., `lru_cache`, `build_py`).

- Streamline scenario management and enhance save functionality
  ([`a4cea07`](https://github.com/jacksonpradolima/coleman/commit/a4cea07e491b095531a643a8be9be63ed182b146))

Scenario Management: - Remove `has_variants` method; inline logic to check scenario variants. #
  Update condition to explicitly check `VirtualHCSScenario` type and its variants.

Save Functionality: + Add `interval` parameter to `save_periodically` for configurable save
  frequency. # Refactor save logic to use dynamic interval instead of a fixed value.

Dependencies: - Remove unused imports from `scenarios.py` (e.g., `lru_cache`, `build_py`).

- Update Codecov workflow to improve Python version handling and dependency installation
  ([`6651e6c`](https://github.com/jacksonpradolima/coleman/commit/6651e6c689a52492275a1cbefb067518605f8870))

- Update README, convert notebook to marimo in notebooks/, add docs content
  ([`96d0964`](https://github.com/jacksonpradolima/coleman/commit/96d096463a85b0d4159ccf9eb933ec6dacd71df6))

Co-authored-by: jacksonpradolima <7774063+jacksonpradolima@users.noreply.github.com>

- **evaluation**: Minor refactoring
  ([`7b14953`](https://github.com/jacksonpradolima/coleman/commit/7b14953ac3a356f7d7a455b1959297327813b6b1))

- **pylint**: Fix .pylintrc path
  ([`557d773`](https://github.com/jacksonpradolima/coleman/commit/557d77323c9fa15ac95fef8254c139c141a55c3e))

- **test-bandit**: Update imports
  ([`5a3e2ee`](https://github.com/jacksonpradolima/coleman/commit/5a3e2eeabc975cd99984f053189e74940580335d))

### Testing

- Add comprehensive tests for main application logic
  ([`a85d92d`](https://github.com/jacksonpradolima/coleman/commit/a85d92d058fb9139488264fa71aa30e8e82b352f))

Testing: + Add unit tests for core functions like logging, agent creation, and CSV merging. + Add
  integration tests for end-to-end workflows and component interactions. + Document tested
  functionalities, including dynamic class loading and scenario setup. + Include dependencies like
  `pytest` and `unittest.mock` for test coverage support.

- Enhance Environment class unit tests
  ([`25f9484`](https://github.com/jacksonpradolima/coleman/commit/25f94846ba532e99da36c8d72bed9d6e6cd2bd5e))

Testing: + Add comprehensive docstring outlining test cases for the `Environment` class. # Fix
  pickle.load mock return value for consistency in `test_load_experiment`.

- Improve floating-point comparison precision in scenario tests
  ([`d92a0aa`](https://github.com/jacksonpradolima/coleman/commit/d92a0aaed8ed83a66df054512a86b7d2e399f7be))

Testing: # Update assertions to use precise floating-point comparison with a tolerance of 1e-6.

- Improve randomness consistency in policy tests
  ([`6fcb5db`](https://github.com/jacksonpradolima/coleman/commit/6fcb5db459ab40a6278856d4fbc92987e493464e))

Policy Tests: # Update action sampling to use consistent RNG method with no replacement.

- Minor improvements
  ([`2cfce08`](https://github.com/jacksonpradolima/coleman/commit/2cfce089a45596b21b82ddf70f21c2a92cce0ae5))

- Refactor imports and add new policy test coverage
  ([`bd2df2a`](https://github.com/jacksonpradolima/coleman/commit/bd2df2a11f2fc0826cae20d5ef225ca1de95692d))

Policy Tests: + Add test coverage for SWLinUCBPolicy's `choose_all` method. - Remove redundant
  imports for LinUCBPolicy, SWLinUCBPolicy, EpsilonGreedyPolicy, GreedyPolicy, UCB1Policy, and
  FRRMABPolicy.

Benchmarking: # Fix formatting in simulation for consistency in test setup.

- Update reward evaluation tests
  ([`9e7e242`](https://github.com/jacksonpradolima/coleman/commit/9e7e24261746bc25ae1b67359d0e8c809628b284))

Testing: - Remove unnecessary trailing newline in `test_reward.py`.

- **agent**: Add module docstring
  ([`852a84e`](https://github.com/jacksonpradolima/coleman/commit/852a84e7df6480d4b91de8d3bb6dc0f119e761c6))

- **bandit**: Add tests for improved coverage and performance validation
  ([`d88d65e`](https://github.com/jacksonpradolima/coleman/commit/d88d65e43d2862265ce801cadb28e1023df51fad))

- Added tests for `update_priority` with large datasets to benchmark and validate performance
  improvements. - Included tests for handling empty and invalid actions in
  `EvaluationMetricBandit.pull`. - Verified the correctness of the priority mapping logic with
  duplicate arm names.

- **evaluation**: Add docstrings
  ([`24f2b3f`](https://github.com/jacksonpradolima/coleman/commit/24f2b3f7158692ad4f33080589af212eb85b9919))

- **evaluation**: Added `RunningEvaluationTests` with comprehensive test cases for NAPFD and
  NAPFDVerdict metrics.
  ([`9570fc7`](https://github.com/jacksonpradolima/coleman/commit/9570fc74951c005677cb7e2d6c5d5b35ac3f8e96))

- **monitor**: Add missing docstring
  ([`dcf9eb4`](https://github.com/jacksonpradolima/coleman/commit/dcf9eb4c007a2c9bf922ca9c2b4884cdde68f09b))

- **monitor**: Add test case for empty `batch_df` in `collect_from_temp`
  ([`4aa0b77`](https://github.com/jacksonpradolima/coleman/commit/4aa0b775423061a8fb934bb59c3dbbafc7c34978))

- Added `test_collect_from_temp_empty_batch_df` to verify `df` remains unchanged when `batch_df` is
  empty or invalid. - Confirmed `temp_rows` is cleared even if `batch_df` is empty.

- **monitor**: Add tests to improve branch coverage for temp_rows and save functionality
  ([`4ac5ad8`](https://github.com/jacksonpradolima/coleman/commit/4ac5ad8a9666cabe6a687e68e940b139f6b60b2e))

- Added `test_save_with_empty_temp_rows` to verify behavior when `temp_rows` is empty during save. -
  Added `test_save_with_non_empty_temp_rows` to ensure data in `temp_rows` is properly flushed and
  saved. - Improved coverage for `if self.temp_rows:` branch in `save` method and related scenarios.

- **monitor**: Created a specific data class to collect the data for monitor
  ([`edd4dda`](https://github.com/jacksonpradolima/coleman/commit/edd4dda7d5da5ed1ff16b10255dca2b439698be3))

- **monitor**: Do not perform equality checks with floating point values.
  ([`ba9bcc7`](https://github.com/jacksonpradolima/coleman/commit/ba9bcc72281463ead8df54056ba9136380bb4f5d))

- **monitor**: Improved docstring
  ([`d83fd9d`](https://github.com/jacksonpradolima/coleman/commit/d83fd9dc4d1b1e2c4dc2d30a888b03a9b6e49f02))

- **monitor**: Minor enhancements
  ([`15e3ca5`](https://github.com/jacksonpradolima/coleman/commit/15e3ca5b81396f6c64336b5c4f73a3642d362264))

- **policy**: Disable policy test
  ([`ff4a4bb`](https://github.com/jacksonpradolima/coleman/commit/ff4a4bb1318aa3400ce078542e98ccaa79166d43))

- **policy**: Removed policy test
  ([`ae4ffdd`](https://github.com/jacksonpradolima/coleman/commit/ae4ffdd8e6ade8c53b2cb2c9dcd9ba23411a93ad))
