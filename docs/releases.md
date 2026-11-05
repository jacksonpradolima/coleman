# Release preparation

Release descriptions explain the user's problem, supported workflows and evidence
before listing internal work. Stable v4 centers on CI test prioritization and
selection. The v5 architecture broadens the independent decision components while
preserving CI/TCPCI as a first-class application; its release notes must describe
the qualified scope rather than promise arbitrary problem solvability.

## Versioning and integration

The next major version is developed on `release/v5`, with one reviewed issue PR
at a time targeting that branch. Integration runs CI without publishing releases.
The [v5 integration guide](v5-release-integration.md) explains the single final
5.0 release, stable v4 maintenance and isolated alpha/beta/RC rehearsals.
The [architecture](v5-architecture.md) and [compatibility inventory](v5-compatibility.md)
describe the intended changes; those specifications do not advertise implemented
generic APIs.

The published [COLEMAN 4.9](releases/4.9.0.md) release adds [larger LRTS histories
with explicit stage views](lrts-stage-views.md): frozen ActiveMQ and Log4j source
profiles, analytical/native Parquet and actual checkpointed replay. Source
groups and temporal blockers remain explicit. The signed wheel, source archive,
SBOM and synchronized lock are available in [the release
assets](https://github.com/jacksonpradolima/coleman/releases/tag/v4.9.0).

The published [COLEMAN 4.8](releases/4.8.0.md) release combines bounded ZIP64
source integrity and the [complete RTPTorrent Parquet/replay workflow](rtptorrent-parquet-etl.md),
including the Linux worker memory-accounting correction. The verified wheel,
source archive, Sigstore bundles, SBOM and synchronized lock are available in
[the release assets](https://github.com/jacksonpradolima/coleman/releases/tag/v4.8.0).

[COLEMAN 4.7](releases/4.7.0.md) introduced the real
[LRTS download-to-Parquet workflow](lrts-parquet-etl.md), including analytical
tables, explicit native simulation input and actual runner/checkpoint recovery.
Its source-admission and confirmatory-evidence limits remain explicit.

[COLEMAN 4.6](releases/4.6.0.md) adds versioned controlled fault observations,
a pinned offline JUnit3 source recipe and recoverable raw evidence. Existing
native TCPCI experiments and learning checkpoints keep their contracts. Controlled
source reproduction remains separate from benchmark admission.

[COLEMAN 4.5](releases/4.5.0.md) added bounded RTPTorrent and LRTS source readers,
explicit units/provenance and installed native CI/checkpoint workflows. It retains
the exact numeric safeguards from 4.4.1, corpus profiles and secure source cache.
Both external sources remain blocked for protected temporal evaluation.
The public corpus program continues in subsequent compatible **4.x** feature releases; compatible corrections use patch
versions. Follow the [public roadmap](roadmap.md) in dependency order and keep each
PR runnable and independently reviewed. Publishing a slice does not complete the
remaining corpus-admission or confirmatory-evidence gates.

A release branch can collect related reviewed PRs when a capability needs several
vertical slices before publication. Keep it synchronized with security fixes from
`main`; available security corrections must not wait for a feature release.
The publication workflow runs on `main`. A compatible `feat:` merge requests the
next minor version; `fix:` requests a patch. Reserve `feat!:` or an explicit
breaking-change declaration for an actual incompatible public contract.

Before integrating a release branch, review its entire diff and pass the full
quality/security/install/native-CI gates. Do not manually edit only the package
version while leaving tags or the lock inconsistent.
The automated workflow updates the package version, synchronizes `uv.lock`, builds
the distributions, and includes the lock in the version commit before tagging.
A second lock check after tagging must find no changes. This keeps the tag, source
archive and wheel metadata consistent.

## Reviewed release descriptions

Before the final release PR, commit `docs/releases/<version>.md` with an en-US release
description. The publication workflow uses `docs/releases/<version>.md` when
present; otherwise it preserves the generated description. These notes are part
of the reviewed diff and must describe only completed, validated public features.

Write for a new CI user:

- the concrete problem the release solves and useful workflows;
- the delivered capabilities and a runnable starting example;
- compatibility, Python requirements and any migration steps;
- correctness/security evidence and its limits;
- known limitations and links to the full changes.

Avoid unsupported algorithm superiority, universal generalization or wall-clock
savings claims. Separate modeled objectives, measured CI outcomes, mathematical
correctness and protected empirical evidence. Negative and inconclusive outcomes
remain results.

After automatic publication, check the rendered GitHub description, assets,
checksums, attestations, SBOM, package version and source/lock consistency. If the
automatic description needs an editorial correction, update it from a reviewed
notes file rather than replacing it with an unverified marketing claim.
